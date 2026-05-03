// src/lib/aqi.js  — AQI helpers shared across components

export const AQI_LEVELS = [
  { max: 50,  label: "Good",                        color: "#22c55e", bg: "bg-aqi-good",      text: "text-aqi-good",      border: "border-aqi-good/30"      },
  { max: 100, label: "Moderate",                    color: "#eab308", bg: "bg-aqi-moderate",  text: "text-aqi-moderate",  border: "border-aqi-moderate/30"  },
  { max: 150, label: "Unhealthy for Sensitive Groups", color: "#f97316", bg: "bg-aqi-sensitive", text: "text-aqi-sensitive", border: "border-aqi-sensitive/30" },
  { max: 200, label: "Unhealthy",                   color: "#ef4444", bg: "bg-aqi-unhealthy", text: "text-aqi-unhealthy", border: "border-aqi-unhealthy/30" },
  { max: 300, label: "Very Unhealthy",              color: "#a855f7", bg: "bg-aqi-very",      text: "text-aqi-very",      border: "border-aqi-very/30"      },
  { max: Infinity, label: "Hazardous",              color: "#991b1b", bg: "bg-aqi-hazardous", text: "text-aqi-hazardous", border: "border-red-900/40"       },
];

export function getLevel(aqi) {
  return AQI_LEVELS.find((l) => aqi <= l.max) ?? AQI_LEVELS[AQI_LEVELS.length - 1];
}

export const POLLUTANT_META = {
  "PM2.5": { unit: "µg/m³", safe: 12,  warn: 35,  danger: 55  },
  "PM10":  { unit: "µg/m³", safe: 54,  warn: 154, danger: 254 },
  "O3":    { unit: "ppb",   safe: 54,  warn: 70,  danger: 85  },
  "CO":    { unit: "ppm",   safe: 4.5, warn: 9.5, danger: 12.5 },
  "SO2":   { unit: "ppb",   safe: 35,  warn: 75,  danger: 185 },
  "NO2":   { unit: "ppb",   safe: 53,  warn: 100, danger: 360 },
  "AQI":   { unit: "",      safe: 50,  warn: 100, danger: 150 },
};

/** Returns "safe" | "warn" | "danger" for a given pollutant value */
export function getSeverity(pollutant, value) {
  const meta = POLLUTANT_META[pollutant];
  if (!meta) return "safe";
  if (value <= meta.safe)  return "safe";
  if (value <= meta.warn)  return "warn";
  return "danger";
}

export const SEVERITY_COLOURS = {
  safe:   { bar: "bg-aqi-good",      text: "text-aqi-good"      },
  warn:   { bar: "bg-aqi-moderate",  text: "text-aqi-moderate"  },
  danger: { bar: "bg-aqi-unhealthy", text: "text-aqi-unhealthy" },
};

// src/components/PollutantCard.jsx
import { getSeverity, SEVERITY_COLOURS, POLLUTANT_META } from "../lib/aqi";

export default function PollutantCard({ name, value, unit }) {
  const severity = getSeverity(name, value);
  const colours  = SEVERITY_COLOURS[severity];
  const meta     = POLLUTANT_META[name] ?? { safe: 1, warn: 2, danger: 3 };

  // Bar fill: 0–100% mapped to 0–danger threshold × 1.5
  const max     = meta.danger * 1.5;
  const fillPct = Math.min((value / max) * 100, 100);

  return (
    <div className="glass p-4 flex flex-col gap-3 animate-slide-up hover:border-white/10 transition-colors">
      {/* Header */}
      <div className="flex items-start justify-between">
        <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
          {name}
        </span>
        <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${colours.text}
                          bg-current/10`}
              style={{ backgroundColor: "transparent" }}>
          <span className={colours.text}>
            {severity === "safe" ? "Safe" : severity === "warn" ? "Moderate" : "High"}
          </span>
        </span>
      </div>

      {/* Value */}
      <div className="flex items-baseline gap-1.5">
        <span className={`text-2xl font-bold tabular-nums ${colours.text}`}>
          {value % 1 === 0 ? value : value.toFixed(2)}
        </span>
        {unit && (
          <span className="text-xs text-slate-500 font-mono">{unit}</span>
        )}
      </div>

      {/* Severity bar */}
      <div className="h-1.5 rounded-full bg-navy-600 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-1000 ${colours.bar}`}
          style={{ width: `${fillPct}%` }}
        />
      </div>
    </div>
  );
}

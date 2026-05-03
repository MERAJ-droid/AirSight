// src/components/AqiGauge.jsx  — animated SVG circular gauge
import { getLevel } from "../lib/aqi";

const RADIUS = 70;
const STROKE = 12;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;
// We show 240° of the circle (leaving a 120° gap at the bottom)
const ARC = CIRCUMFERENCE * (240 / 360);

function aqiToOffset(aqi) {
  // AQI 0–500 mapped onto the 240° arc
  const pct = Math.min(aqi / 500, 1);
  return ARC - pct * ARC;
}

export default function AqiGauge({ aqi, category }) {
  const level = getLevel(aqi);
  const offset = aqiToOffset(aqi);
  const size = RADIUS * 2 + STROKE * 2 + 4;
  const cx = size / 2;
  const cy = size / 2;

  // Rotate so the arc starts at -210° (bottom-left) and ends at 30° (bottom-right)
  const rotation = 150; // degrees

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative">
        <svg
          width={size}
          height={size}
          viewBox={`0 0 ${size} ${size}`}
          style={{ transform: `rotate(${rotation}deg)` }}
        >
          {/* Background track */}
          <circle
            cx={cx} cy={cy} r={RADIUS}
            fill="none"
            stroke="#1e2a47"
            strokeWidth={STROKE}
            strokeDasharray={`${ARC} ${CIRCUMFERENCE - ARC}`}
            strokeLinecap="round"
          />
          {/* Value arc */}
          <circle
            cx={cx} cy={cy} r={RADIUS}
            fill="none"
            stroke={level.color}
            strokeWidth={STROKE}
            strokeDasharray={`${ARC} ${CIRCUMFERENCE - ARC}`}
            strokeDashoffset={offset}
            strokeLinecap="round"
            className="gauge-ring"
            style={{ filter: `drop-shadow(0 0 8px ${level.color}80)` }}
          />
        </svg>

        {/* Centre text */}
        <div
          className="absolute inset-0 flex flex-col items-center justify-center"
          style={{ transform: `rotate(-${rotation}deg)` }}
        >
          <span
            className="text-4xl font-bold tabular-nums leading-none"
            style={{ color: level.color }}
          >
            {Math.round(aqi)}
          </span>
          <span className="text-xs text-slate-500 mt-1 font-medium">AQI</span>
        </div>
      </div>

      {/* Category label */}
      <span
        className="text-sm font-semibold px-3 py-1 rounded-full border"
        style={{
          color: level.color,
          borderColor: `${level.color}40`,
          backgroundColor: `${level.color}15`,
        }}
      >
        {category}
      </span>
    </div>
  );
}

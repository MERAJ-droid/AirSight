// src/App.jsx  — root layout + state orchestration
import { useState, useCallback } from "react";
import { Satellite, Activity, AlertCircle } from "lucide-react";

import UploadZone    from "./components/UploadZone";
import AqiGauge      from "./components/AqiGauge";
import PollutantCard from "./components/PollutantCard";
import ChatPanel     from "./components/ChatPanel";

import { predict, analyze } from "./lib/api";
import { POLLUTANT_META }   from "./lib/aqi";

// Pollutants to show in the grid (all except AQI which gets the gauge)
const POLLUTANTS = Object.keys(POLLUTANT_META).filter((k) => k !== "AQI");

export default function App() {
  const [loading,    setLoading]    = useState(false);
  const [advisory,   setAdvisory]   = useState(null);   // string
  const [sources,    setSources]    = useState([]);
  const [result,     setResult]     = useState(null);   // full /predict response
  const [error,      setError]      = useState(null);

  const handleFile = useCallback(async (file) => {
    if (!file) {
      setResult(null);
      setAdvisory(null);
      setSources([]);
      setError(null);
      return;
    }

    setLoading(true);
    setResult(null);
    setAdvisory(null);
    setSources([]);
    setError(null);

    try {
      // 1. Fast CNN predictions — show results immediately
      const fast = await predict(file);
      setResult(fast);

      // 2. Full RAG advisory — arrives shortly after
      const full = await analyze(file);
      setAdvisory(full.advisory);
      setSources(full.sources ?? []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  return (
    <div className="min-h-screen flex flex-col bg-navy-950">
      {/* ── Top nav ── */}
      <header className="flex items-center gap-3 px-6 py-4 border-b border-white/5 backdrop-blur-sm bg-navy-950/80 sticky top-0 z-10">
        <div className="flex items-center gap-2.5">
          <div className="p-1.5 rounded-lg bg-blue-500/10 border border-blue-500/20">
            <Satellite className="w-5 h-5 text-blue-400" />
          </div>
          <span className="text-lg font-bold text-white tracking-tight">AirSight</span>
        </div>
        <span className="text-xs text-slate-600 ml-1 hidden sm:block">
          AI-powered air quality analysis
        </span>
        <div className="ml-auto flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-aqi-good animate-pulse-slow" />
          <span className="text-xs text-slate-500">API connected</span>
        </div>
      </header>

      {/* ── Main 3-column layout ── */}
      <main className="flex-1 grid grid-cols-1 lg:grid-cols-[300px_1fr_360px] gap-4 p-4 min-h-0">

        {/* ── LEFT: Upload ── */}
        <aside className="glass p-5 flex flex-col gap-5">
          <div>
            <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
              Image Input
            </h2>
            <UploadZone onFile={handleFile} loading={loading} />
          </div>

          {/* Model info */}
          <div className="mt-auto">
            <div className="rounded-xl border border-white/5 bg-navy-700/30 p-3 flex flex-col gap-2">
              <div className="flex items-center gap-2">
                <Activity className="w-3.5 h-3.5 text-blue-400" />
                <span className="text-xs font-medium text-slate-400">Model Info</span>
              </div>
              <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                {[
                  ["Architecture", "ResNet-50"],
                  ["Outputs",      "7 pollutants"],
                  ["Test R²",      "0.9475"],
                  ["Samples",      "1,061"],
                ].map(([k, v]) => (
                  <div key={k}>
                    <p className="text-[10px] text-slate-600">{k}</p>
                    <p className="text-xs text-slate-300 font-medium font-mono">{v}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </aside>

        {/* ── CENTRE: Results ── */}
        <section className="flex flex-col gap-4 min-h-0 overflow-y-auto">

          {/* Error */}
          {error && (
            <div className="glass p-4 flex items-start gap-3 border-red-500/20 animate-fade-in">
              <AlertCircle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
              <div>
                <p className="text-sm font-medium text-red-400">Analysis failed</p>
                <p className="text-xs text-slate-500 mt-0.5">{error}</p>
              </div>
            </div>
          )}

          {/* Gauge + cards */}
          {(result || loading) && (
            <div className="glass p-6 animate-fade-in">
              <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-6">
                Air Quality Readings
              </h2>

              {result ? (
                <>
                  {/* AQI gauge */}
                  <div className="flex justify-center mb-6">
                    <AqiGauge aqi={result.aqi} category={result.category} />
                  </div>

                  {/* Pollutant grid */}
                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                    {POLLUTANTS.map((p) => (
                      <PollutantCard
                        key={p}
                        name={p}
                        value={result.predictions[p]}
                        unit={result.units?.[p] ?? ""}
                      />
                    ))}
                  </div>

                  {/* Timing */}
                  <p className="text-[10px] text-slate-600 text-right mt-4 font-mono">
                    CNN inference: {result.processing_time_ms.toFixed(0)} ms
                  </p>
                </>
              ) : (
                /* Skeleton */
                <div className="flex flex-col gap-4">
                  <div className="flex justify-center">
                    <div className="skeleton w-40 h-40 rounded-full" />
                  </div>
                  <div className="grid grid-cols-3 gap-3">
                    {[...Array(6)].map((_, i) => (
                      <div key={i} className="skeleton h-24" />
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Health Advisory */}
          {(advisory || (result && !advisory && loading)) && (
            <div className="glass p-6 animate-slide-up">
              <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-4">
                Health Advisory
              </h2>
              {advisory ? (
                <>
                  <div className="text-sm text-slate-300 leading-relaxed whitespace-pre-wrap">
                    {advisory}
                  </div>
                  {sources.length > 0 && (
                    <div className="mt-4 pt-4 border-t border-white/5">
                      <p className="text-[10px] text-slate-600 mb-2">Sources</p>
                      <div className="flex flex-wrap gap-2">
                        {sources.map((s) => (
                          <span key={s} className="pill bg-navy-700 border border-white/5 text-slate-500">
                            📄 {s}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <div className="flex flex-col gap-2">
                  <div className="skeleton h-4 w-full" />
                  <div className="skeleton h-4 w-5/6" />
                  <div className="skeleton h-4 w-4/6" />
                  <div className="skeleton h-4 w-full" />
                </div>
              )}
            </div>
          )}

          {/* Empty state */}
          {!result && !loading && !error && (
            <div className="glass flex-1 flex flex-col items-center justify-center gap-4 py-20 animate-fade-in">
              <div className="p-4 rounded-full bg-blue-500/10 border border-blue-500/10">
                <Satellite className="w-10 h-10 text-blue-400/50" />
              </div>
              <div className="text-center">
                <p className="text-slate-400 font-medium">No analysis yet</p>
                <p className="text-slate-600 text-sm mt-1">
                  Upload an air pollution image to begin
                </p>
              </div>
            </div>
          )}
        </section>

        {/* ── RIGHT: Chat ── */}
        <aside className="glass flex flex-col min-h-0 overflow-hidden">
          <ChatPanel predictions={result?.predictions ?? null} />
        </aside>
      </main>
    </div>
  );
}

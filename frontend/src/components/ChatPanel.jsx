// src/components/ChatPanel.jsx
import { useState, useRef, useEffect } from "react";
import { Send, Bot, User, BookOpen, Loader2 } from "lucide-react";
import { chat } from "../lib/api";

export default function ChatPanel({ predictions }) {
  const [messages, setMessages] = useState([
    {
      role: "ai",
      text: "Hi! I'm the AirSight assistant. Upload an image to analyse air quality, then ask me anything about the results or air quality in general.",
      sources: [],
    },
  ]);
  const [input, setInput]   = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);

  // Scroll to bottom whenever messages update
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function send() {
    const q = input.trim();
    if (!q || loading) return;
    setInput("");
    setMessages((prev) => [...prev, { role: "user", text: q, sources: [] }]);
    setLoading(true);
    try {
      const data = await chat(q, predictions);
      setMessages((prev) => [
        ...prev,
        { role: "ai", text: data.answer, sources: data.sources ?? [] },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "ai", text: `⚠ Error: ${err.message}`, sources: [] },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function handleKey(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  // Suggested questions (shown when predictions exist)
  const suggestions = predictions
    ? ["Should I wear a mask today?", "Is it safe to exercise outdoors?", "Who is most at risk?"]
    : ["What is AQI?", "Which pollutants are most harmful?", "How is PM2.5 measured?"];

  return (
    <div className="flex flex-col h-full min-h-0">
      {/* Header */}
      <div className="flex items-center gap-2 px-4 py-3 border-b border-white/5">
        <Bot className="w-4 h-4 text-blue-400" />
        <span className="text-sm font-semibold text-slate-200">AirSight Chat</span>
        {predictions && (
          <span className="ml-auto pill bg-blue-500/10 border border-blue-500/20 text-blue-400">
            Context loaded
          </span>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-4 flex flex-col gap-3 min-h-0">
        {messages.map((msg, i) => (
          <div key={i} className={`flex flex-col gap-1.5 ${msg.role === "user" ? "items-end" : "items-start"}`}>
            {/* Role icon */}
            <div className="flex items-center gap-1.5">
              {msg.role === "ai"
                ? <Bot className="w-3.5 h-3.5 text-blue-400" />
                : <User className="w-3.5 h-3.5 text-slate-500" />}
              <span className="text-[10px] text-slate-500">
                {msg.role === "ai" ? "AirSight AI" : "You"}
              </span>
            </div>

            {/* Bubble */}
            <div className={msg.role === "user" ? "bubble-user" : "bubble-ai"}>
              <p className="leading-relaxed whitespace-pre-wrap text-slate-200">{msg.text}</p>
            </div>

            {/* Sources */}
            {msg.sources && msg.sources.length > 0 && (
              <div className="flex flex-wrap gap-1.5 mt-1">
                {msg.sources.map((s) => (
                  <span key={s} className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full
                                           bg-navy-700 border border-white/5 text-[10px] text-slate-500">
                    <BookOpen className="w-2.5 h-2.5" />
                    {s}
                  </span>
                ))}
              </div>
            )}
          </div>
        ))}

        {/* Typing indicator */}
        {loading && (
          <div className="flex items-center gap-2 bubble-ai w-fit animate-fade-in">
            <Loader2 className="w-3.5 h-3.5 text-blue-400 animate-spin" />
            <span className="text-xs text-slate-500">Thinking…</span>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Suggestions */}
      {!loading && (
        <div className="px-4 pb-2 flex flex-wrap gap-1.5">
          {suggestions.map((s) => (
            <button
              key={s}
              onClick={() => { setInput(s); }}
              className="text-[10px] px-2.5 py-1 rounded-full border border-white/5
                         bg-navy-700/50 text-slate-500 hover:text-slate-300 hover:border-white/10
                         transition-colors truncate max-w-[160px]"
            >
              {s}
            </button>
          ))}
        </div>
      )}

      {/* Input */}
      <div className="px-4 pb-4">
        <div className="flex gap-2 items-end border border-white/10 rounded-xl bg-navy-700/50
                        focus-within:border-blue-500/40 transition-colors p-2">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKey}
            placeholder="Ask about air quality…"
            rows={1}
            className="flex-1 bg-transparent text-sm text-slate-200 placeholder-slate-600
                       resize-none outline-none leading-relaxed min-h-[24px] max-h-[120px]"
          />
          <button
            onClick={send}
            disabled={!input.trim() || loading}
            className="p-2 rounded-lg bg-blue-600 hover:bg-blue-500 disabled:opacity-40
                       disabled:pointer-events-none transition-colors shrink-0"
          >
            <Send className="w-3.5 h-3.5 text-white" />
          </button>
        </div>
        <p className="text-[10px] text-slate-600 mt-1.5 text-center">
          Enter to send · Shift+Enter for new line
        </p>
      </div>
    </div>
  );
}

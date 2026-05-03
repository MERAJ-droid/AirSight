// src/lib/api.js  — thin wrapper around the AirSight FastAPI backend

const BASE = "http://localhost:8000";

/**
 * POST /predict  — CNN only, returns predictions fast
 * @param {File} imageFile
 * @returns {Promise<object>}
 */
export async function predict(imageFile) {
  const form = new FormData();
  form.append("file", imageFile);
  const res = await fetch(`${BASE}/predict`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`predict: ${res.status} ${await res.text()}`);
  return res.json();
}

/**
 * POST /analyze  — full pipeline: CNN + FAISS + Gemini
 * @param {File} imageFile
 * @returns {Promise<object>}
 */
export async function analyze(imageFile) {
  const form = new FormData();
  form.append("file", imageFile);
  const res = await fetch(`${BASE}/analyze`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`analyze: ${res.status} ${await res.text()}`);
  return res.json();
}

/**
 * POST /chat  — RAG-grounded Q&A
 * @param {string} question
 * @param {object|null} predictions  — optional context from /analyze
 * @returns {Promise<object>}
 */
export async function chat(question, predictions = null) {
  const res = await fetch(`${BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, predictions }),
  });
  if (!res.ok) throw new Error(`chat: ${res.status} ${await res.text()}`);
  return res.json();
}

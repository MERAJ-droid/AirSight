"""
api/main.py — AirSight FastAPI backend.

Exposes three endpoints:
  GET  /health   → server + resource liveness check
  POST /predict  → CNN inference only  (fast, ~1-3 s)
  POST /analyze  → CNN + FAISS + Gemini advisory  (~5-10 s)

Run from the project root with the WSL venv active:
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Then open http://localhost:8000/docs in your Windows browser to test
with the built-in Swagger UI (supports image file uploads).
"""

from __future__ import annotations

import os
import sys
import pickle
import time
import tempfile
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

# ── Silence verbose logs before heavy imports ─────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

# ── Rate limiting ─────────────────────────────────────────────────────────────
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.requests import Request

import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.startup import download_artifacts_if_needed

# ── Local imports ─────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR   = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from model import TARGET_COLS  # noqa: E402

from api.schemas import (  # noqa: E402
    AnalysisResponse,
    PredictionResponse,
    HealthResponse,
    ChatRequest,
    ChatResponse,
    POLLUTANT_UNITS,
)

import google.generativeai as genai  # noqa: E402
from dotenv import load_dotenv        # noqa: E402
from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402
from langchain_community.vectorstores import FAISS        # noqa: E402

load_dotenv(REPO_ROOT / ".env")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH  = REPO_ROOT / "artifacts" / "best_model.keras"
SCALER_PATH = REPO_ROOT / "artifacts" / "scaler.pkl"
STORE_PATH  = str(REPO_ROOT / "artifacts" / "vectorstore")
IMG_SIZE    = (224, 224)

# ── AQI helpers ───────────────────────────────────────────────────────────────

def get_aqi_category(aqi: float) -> str:
    if aqi <= 50:  return "Good"
    if aqi <= 100: return "Moderate"
    if aqi <= 150: return "Unhealthy for Sensitive Groups"
    if aqi <= 200: return "Unhealthy"
    if aqi <= 300: return "Very Unhealthy"
    return "Hazardous"


# ── Image preprocessing ───────────────────────────────────────────────────────

def preprocess_image(image_path: str) -> np.ndarray:
    """Load, resize and normalise a single image for ResNet50."""
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img)
    arr = tf.keras.applications.resnet50.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)   # (1, 224, 224, 3)


# ── Inference helper ──────────────────────────────────────────────────────────

def run_inference(model, scaler, image_path: str) -> dict[str, float]:
    """
    Run a single image through the CNN and inverse-scale the outputs.

    Returns a dict mapping pollutant name → real-world value (float, 2 d.p.)
    """
    img_array   = preprocess_image(image_path)
    raw_output  = model.predict(img_array, verbose=0)

    # raw_output may be a dict (multi-output) or a list/array
    if isinstance(raw_output, dict):
        scaled = np.array([[raw_output[col][0][0] for col in TARGET_COLS]])
    else:
        scaled = np.array([[raw_output[i][0][0] for i in range(len(TARGET_COLS))]])

    real = scaler.inverse_transform(scaled)[0]   # shape (7,)
    return {col: round(float(real[i]), 2) for i, col in enumerate(TARGET_COLS)}


# ── RAG advisory helper ───────────────────────────────────────────────────────

def generate_advisory(
    predictions: dict[str, float],
    vectorstore,          # pre-loaded FAISS vectorstore
) -> tuple[str, list[str]]:
    """
    Query the pre-loaded FAISS vectorstore and call Gemini for the advisory.
    Mirrors rag_explainer.explain_with_rag() but reuses the already-loaded store.
    """
    worst_pollutant = max(
        {k: v for k, v in predictions.items() if k != "AQI"}, key=predictions.get
    )
    query = (
        f"Health effects and safety guidelines for high levels of "
        f"{worst_pollutant} and AQI {predictions['AQI']}"
    )

    docs    = vectorstore.similarity_search(query, k=3)
    context = "\n---\n".join([d.page_content for d in docs])
    # Use Path to normalise both forward- and back-slash paths from metadata
    sources = sorted(
        {Path(d.metadata.get("source", "Unknown")).name for d in docs}
    )

    prompt = f"""
    You are the AirSight AI Assistant. You analyzed an image and predicted:
    {predictions}

    Official air quality guideline context:
    {context}

    Using ONLY the metrics above and the provided context, write a professional
    health advisory. Mention specific pollutants and explain the risks based on
    the retrieved guidelines. Keep it concise but authoritative.
    """

    try:
        gemini = genai.GenerativeModel("gemini-2.5-flash")
        response = gemini.generate_content(prompt)
        return response.text, sources
    except Exception as exc:
        return f"Advisory generation failed: {exc}", []


# ── Lifespan — load all heavy resources once at startup ───────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Resources loaded here are kept in memory for the lifetime of the server.
    This avoids re-loading the model and vectorstore on every request.
    """
    print("\n🛰  AirSight API  ·  Starting up ...")

    # Download model artifacts from private HF Hub repo (HF Spaces deployment).
    # No-op when running locally (artifacts/ already exists).
    download_artifacts_if_needed(REPO_ROOT / "artifacts")

    # CNN model
    print(f"  ⚙  Loading model  →  {MODEL_PATH.name}")
    app.state.model = tf.keras.models.load_model(str(MODEL_PATH))

    # Scaler
    print(f"  ⚙  Loading scaler →  {SCALER_PATH.name}")
    with open(SCALER_PATH, "rb") as f:
        app.state.scaler = pickle.load(f)

    # Sentence-Transformers embedding model (CPU)
    print("  ⚙  Loading embeddings  →  all-MiniLM-L6-v2  (CPU)")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    # FAISS vectorstore
    print(f"  ⚙  Loading vectorstore  →  {STORE_PATH}")
    app.state.vectorstore = FAISS.load_local(
        STORE_PATH, embeddings, allow_dangerous_deserialization=True
    )

    print("  ✅  All resources loaded.  Listening ...\n")
    app.state.ready = True

    yield   # ← server runs here

    # Shutdown (nothing to clean up explicitly)
    print("\n🛰  AirSight API  ·  Shutting down.")
    app.state.ready = False


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AirSight API",
    description=(
        "Air Quality Estimation from Images.\n\n"
        "Upload an aerial/street-level photo → get pollutant predictions (PM2.5, "
        "PM10, O3, CO, SO2, NO2, AQI) and an AI-generated health advisory grounded "
        "in official EPA/CPCB guidelines."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── Rate limiter ─────────────────────────────────────────────────────────────
# Per-IP limits prevent Gemini API quota abuse from the public HF Space.
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Locked to the production Vercel domain + local dev. No wildcard in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://airsight.vercel.app",   # production frontend
        "http://localhost:5173",          # local Vite dev server
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness check",
    tags=["Meta"],
)
async def health():
    """Returns server status and whether the model/vectorstore are loaded."""
    return HealthResponse(
        status="ok",
        model_loaded=hasattr(app.state, "model"),
        vectorstore_loaded=hasattr(app.state, "vectorstore"),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="CNN inference only  (fast)",
    tags=["Inference"],
)
@limiter.limit("10/minute")
async def predict(request: Request, file: UploadFile = File(..., description="Air pollution image (JPG/PNG)")):
    """
    Upload an image → returns CNN pollutant predictions in real-world units.

    - Runs **only the CNN** (no LLM call).
    - Typical latency: **1–3 seconds** on GPU.
    """
    _validate_image(file)
    t0 = time.perf_counter()

    with _save_temp(file) as tmp_path:
        predictions = run_inference(app.state.model, app.state.scaler, tmp_path)

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    return PredictionResponse(
        aqi=predictions["AQI"],
        category=get_aqi_category(predictions["AQI"]),
        predictions=predictions,
        units=POLLUTANT_UNITS,
        processing_time_ms=elapsed_ms,
    )


@app.post(
    "/analyze",
    response_model=AnalysisResponse,
    summary="Full pipeline: CNN + RAG + Gemini advisory",
    tags=["Inference"],
)
@limiter.limit("5/minute")
async def analyze(request: Request, file: UploadFile = File(..., description="Air pollution image (JPG/PNG)")):
    """
    Upload an image → returns pollutant predictions **and** an AI health advisory.

    - Runs the **full pipeline**: CNN → FAISS retrieval → Gemini.
    - Typical latency: **5–10 seconds** (network call to Gemini included).
    - The advisory is grounded strictly in official PDF documents (EPA/CPCB).
    """
    _validate_image(file)
    t0 = time.perf_counter()

    with _save_temp(file) as tmp_path:
        predictions = run_inference(app.state.model, app.state.scaler, tmp_path)

    advisory, sources = generate_advisory(predictions, app.state.vectorstore)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    return AnalysisResponse(
        aqi=predictions["AQI"],
        category=get_aqi_category(predictions["AQI"]),
        predictions=predictions,
        units=POLLUTANT_UNITS,
        processing_time_ms=elapsed_ms,
        advisory=advisory,
        sources=sources,
    )


@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="Ask a question about air quality (RAG-grounded)",
    tags=["Chat"],
)
@limiter.limit("5/minute")
async def chat(request: Request, body: ChatRequest):
    """
    Ask any air quality question — answered using the AirSight knowledge base.

    **Tip:** Pass the `predictions` dict from a previous `/analyze` call so
    Gemini can give advice specific to *that* reading rather than a generic answer.

    Example questions:
    - *"What are the long-term health effects of high PM2.5 exposure?"*
    - *"Should I wear a mask with AQI 186?"*
    - *"Which pollutants are most dangerous for children?"*
    """
    t0 = time.perf_counter()

    # Retrieve relevant chunks
    docs    = app.state.vectorstore.similarity_search(body.question, k=4)
    context = "\n---\n".join([d.page_content for d in docs])
    sources = sorted({Path(d.metadata.get("source", "Unknown")).name for d in docs})

    # Build context-aware prompt
    context_block = ""
    if body.predictions:
        context_block = (
            f"\nThe user's current air quality readings are:\n{body.predictions}\n"
            "Use these readings to make your answer specific and actionable.\n"
        )

    prompt = f"""
    You are the AirSight AI Assistant — an expert in air quality and environmental health.
    Answer the user's question using ONLY the provided official guideline context.
    {context_block}
    Official guideline context:
    {context}

    User question: {body.question}

    Answer clearly, professionally, and in plain language. Do not introduce information
    outside the provided context.
    """

    try:
        gemini = genai.GenerativeModel("gemini-2.5-flash")
        response = gemini.generate_content(prompt)
        answer = response.text
    except Exception as exc:
        answer = f"Chat error: {exc}"

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    return ChatResponse(
        answer=answer,
        sources=sources,
        processing_time_ms=elapsed_ms,
    )


# ── Private helpers ───────────────────────────────────────────────────────────

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}


def _validate_image(file: UploadFile):
    """Raise 400 if the uploaded file is not a recognised image type."""
    ct = (file.content_type or "").lower()
    if ct not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ct}'. "
                   f"Accepted: {sorted(ALLOWED_CONTENT_TYPES)}",
        )


class _save_temp:
    """Context manager: saves UploadFile bytes to a temp file, yields path, deletes on exit."""

    def __init__(self, upload: UploadFile):
        self._upload = upload
        self._path: str = ""

    def __enter__(self) -> str:
        suffix = Path(self._upload.filename or "image.jpg").suffix or ".jpg"
        fd, self._path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        # Read bytes synchronously (fine for our single-worker model)
        contents = self._upload.file.read()
        with open(self._path, "wb") as f:
            f.write(contents)
        return self._path

    def __exit__(self, *_):
        try:
            os.remove(self._path)
        except OSError:
            pass

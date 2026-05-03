# ── Base image ────────────────────────────────────────────────────────────────
# python:3.12-slim keeps the image lean while matching the dev venv Python version
FROM python:3.12-slim

# HF Spaces runs containers as a non-root user — workdir must be world-readable
WORKDIR /app

# ── System dependencies ───────────────────────────────────────────────────────
# Required by TensorFlow (OpenMP) and Pillow (image libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy requirements first so Docker can cache this layer separately
COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

# ── Source code ───────────────────────────────────────────────────────────────
# api/ contains the FastAPI app and startup download logic
# src/ contains model.py (TARGET_COLS import) and the RAG/LLM modules
# src/rag/documents/ contains the PDFs used to build the FAISS vectorstore
# NOTE: artifacts/ is intentionally NOT copied — downloaded from HF Hub at runtime
COPY api/  ./api/
COPY src/   ./src/

# ── HF Spaces listens on port 7860 by default ────────────────────────────────
EXPOSE 7860

# ── Startup command ───────────────────────────────────────────────────────────
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]

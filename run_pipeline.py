"""
run_pipeline.py — AirSight end-to-end pipeline.

Chains the Vision CNN with the RAG/LLM health advisory in a single command.
Run this from the repo root in your WSL environment (GPU side).

Usage
-----
    # Full pipeline: CNN prediction + RAG-grounded advisory (recommended)
    python run_pipeline.py --image path/to/image.jpg

    # Quick mode: CNN prediction + direct Gemini advisory (no FAISS lookup)
    python run_pipeline.py --image path/to/image.jpg --mode quick

Data Flow
---------
    Image (JPEG)
        → preprocess_image()           [resize, ResNet50 normalize]
        → model.predict()              [artifacts/best_model.keras]
        → scaler.inverse_transform()   [artifacts/scaler.pkl]
        → predictions dict             {PM2.5, PM10, O3, CO, SO2, NO2, AQI}
        → explain_with_rag()           [FAISS lookup → Gemini 2.5 Flash]
        → health advisory text + cited sources
"""

import argparse
import os
import sys
import pickle
import warnings
from pathlib import Path

# ── Suppress noisy logs before heavy imports ─────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

# ── Repo layout ──────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent
SRC_DIR     = REPO_ROOT / "src"

# Add src/ so we can do `from model import ...` and `from llm.xxx import ...`
sys.path.insert(0, str(SRC_DIR))
# Also add repo root so src.llm.prompts works inside explainer.py
sys.path.insert(1, str(REPO_ROOT))

# ── Artifact paths (absolute — safe on both Windows and WSL) ─────────────────
MODEL_PATH  = REPO_ROOT / "artifacts" / "best_model.keras"
SCALER_PATH = REPO_ROOT / "artifacts" / "scaler.pkl"
STORE_PATH  = str(REPO_ROOT / "artifacts" / "vectorstore")

IMG_SIZE    = (224, 224)

# ── Lazy imports from src (after sys.path is set) ────────────────────────────
from model import TARGET_COLS                                    # noqa: E402
from llm.explainer import get_health_advisory, get_aqi_category  # noqa: E402
from llm.rag_explainer import explain_with_rag                   # noqa: E402

# ── ANSI helpers ─────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
AQI_COLORS = {
    "Good":                          "\033[92m",   # bright green
    "Moderate":                      "\033[93m",   # bright yellow
    "Unhealthy for Sensitive Groups": "\033[33m",  # dark yellow
    "Unhealthy":                     "\033[91m",   # bright red
    "Very Unhealthy":                "\033[35m",   # magenta
    "Hazardous":                     "\033[31m",   # dark red
}
POLLUTANT_UNITS = {
    "PM2.5": "µg/m³", "PM10": "µg/m³",
    "O3":    "ppb",   "CO":   "ppm",
    "SO2":   "ppb",   "NO2":  "ppb",
    "AQI":   "—",
}

W = 54  # banner width


# ─────────────────────────────────────────────────────────────────────────────
# Resource loading
# ─────────────────────────────────────────────────────────────────────────────

def load_resources():
    """Load the Keras model and sklearn scaler. Called once at startup."""
    print(f"\n{DIM}⚙  Loading model  →  {MODEL_PATH.name}{RESET}")
    if not MODEL_PATH.exists():
        _die(f"Model not found at {MODEL_PATH}\n    Run train.py first.")
    model = tf.keras.models.load_model(str(MODEL_PATH))

    print(f"{DIM}⚙  Loading scaler →  {SCALER_PATH.name}{RESET}")
    if not SCALER_PATH.exists():
        _die(f"Scaler not found at {SCALER_PATH}\n    Run train.py first.")
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


# ─────────────────────────────────────────────────────────────────────────────
# Vision inference
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_image(image_path: str) -> tf.Tensor:
    """Read a JPEG, apply ResNet50 preprocessing, return a (1, H, W, 3) tensor."""
    raw   = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(raw, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.keras.applications.resnet50.preprocess_input(image)
    image = tf.expand_dims(image, axis=0)
    return image


def run_vision(model, scaler, image_path: str) -> dict:
    """
    Run CNN inference and inverse-transform to real-world units.

    Returns
    -------
    dict  {pollutant_name: float}  e.g. {"PM2.5": 167.5, "AQI": 186.4, ...}
    """
    tensor   = preprocess_image(image_path)
    raw_pred = model.predict(tensor, verbose=0)

    # Multi-head output is a dict when the model has named heads
    scaled = []
    for col in TARGET_COLS:
        if isinstance(raw_pred, dict):
            val = float(raw_pred[col][0][0])
        else:
            val = float(raw_pred[TARGET_COLS.index(col)][0][0])
        scaled.append(val)

    real = scaler.inverse_transform(np.array(scaled).reshape(1, -1))[0]
    return dict(zip(TARGET_COLS, real.tolist()))


# ─────────────────────────────────────────────────────────────────────────────
# Terminal output helpers
# ─────────────────────────────────────────────────────────────────────────────

def _die(msg: str):
    print(f"\n❌  {msg}")
    sys.exit(1)


def _banner(title: str):
    print(f"\n{'═' * W}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{'═' * W}")


def print_predictions(predictions: dict):
    """Pretty-print the pollutant table with colour-coded AQI category."""
    aqi      = predictions["AQI"]
    category = get_aqi_category(aqi)
    color    = AQI_COLORS.get(category, "")

    _banner("🛰  AirSight  ·  Vision Predictions")
    print(f"  {'Pollutant':<12}  {'Predicted':>12}  Unit")
    print(f"  {'─' * 12}  {'─' * 12}  {'─' * 6}")
    for col, val in predictions.items():
        unit = POLLUTANT_UNITS.get(col, "")
        # Highlight the dominant pollutant (highest non-AQI value)
        print(f"  {col:<12}  {val:>12.2f}  {unit}")

    print(f"{'─' * W}")
    print(
        f"  {BOLD}AQI Category{RESET}  :  "
        f"{color}{BOLD}{category}{RESET}"
    )
    print(f"{'═' * W}")


def print_advisory(advisory: str, sources: list, mode: str):
    """Print the LLM-generated health advisory with source citations."""
    _banner("📋  Health Advisory")
    if mode == "full" and sources:
        src_str = "  |  ".join(sources)
        print(f"  {DIM}📚 Grounded in: {src_str}{RESET}\n")
    print(advisory.strip())
    print(f"{'═' * W}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AirSight — Image to Health Advisory (end-to-end)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_pipeline.py --image data/raw/mumbai.jpg\n"
            "  python run_pipeline.py --image data/raw/mumbai.jpg --mode quick\n"
        ),
    )
    parser.add_argument(
        "--image", required=True,
        help="Path to the input image (JPEG / PNG).",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "quick"],
        default="full",
        help=(
            "'full'  — RAG retrieval from FAISS + Gemini 2.5 Flash (default).\n"
            "'quick' — Direct Gemini advisory, no FAISS lookup."
        ),
    )
    args = parser.parse_args()

    # ── Validate image path ───────────────────────────────────────────────────
    image_path = str(Path(args.image).resolve())
    if not Path(image_path).exists():
        _die(f"Image not found: {image_path}")

    # ── Header ────────────────────────────────────────────────────────────────
    print(f"\n{'═' * W}")
    print(f"{BOLD}  🛰  AirSight Pipeline  ·  mode = {args.mode}{RESET}")
    print(f"  {DIM}Image : {args.image}{RESET}")
    print(f"{'═' * W}")

    # ── Step 1: Load resources (model + scaler) ───────────────────────────────
    model, scaler = load_resources()

    # ── Step 2: Vision inference ──────────────────────────────────────────────
    print(f"\n{DIM}🔭  Running CNN inference ...{RESET}")
    try:
        predictions = run_vision(model, scaler, image_path)
    except Exception as exc:
        _die(f"Inference failed: {exc}")

    print_predictions(predictions)

    # ── Step 3: LLM advisory ──────────────────────────────────────────────────
    if args.mode == "full":
        print(f"\n{DIM}🔍  Querying knowledge base (FAISS) ...{RESET}")
        advisory, sources = explain_with_rag(predictions, store_path=STORE_PATH)
    else:
        print(f"\n{DIM}⚡  Generating quick advisory (no retrieval) ...{RESET}")
        advisory = get_health_advisory(predictions)
        sources  = []

    print_advisory(advisory, sources, mode=args.mode)


if __name__ == "__main__":
    main()

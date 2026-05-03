"""
predict.py — Standalone inference script for AirSight.

Usage:
    python predict.py --image path/to/image.jpg
"""

import argparse
import os
import sys
import pickle
from pathlib import Path

# Silence verbose TensorFlow logs before importing TF
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

# Optional: Match the mixed precision policy used during training
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# ── Paths & Setup ─────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# Now we can import from src
from model import TARGET_COLS

MODEL_PATH  = REPO_ROOT / "artifacts" / "best_model.keras"
SCALER_PATH = REPO_ROOT / "artifacts" / "scaler.pkl"
IMG_SIZE    = (224, 224)


def load_and_preprocess_image(image_path: str) -> tf.Tensor:
    """Read, decode, resize, preprocess, and expand dims for inference."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

    # Read and decode
    raw = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(raw, channels=3)
    
    # Resize to the input shape expected by ResNet50
    image = tf.image.resize(image, IMG_SIZE)
    
    # Apply ResNet50 specific preprocessing (BGR conversion, zero-centering)
    image = tf.keras.applications.resnet50.preprocess_input(image)
    
    # Expand dims to create a batch of 1: (H, W, C) -> (1, H, W, C)
    image = tf.expand_dims(image, axis=0)
    
    return image


def main():
    parser = argparse.ArgumentParser(description="AirSight single image inference")
    parser.add_argument("--image", type=str, required=True, help="Path to the image file")
    args = parser.parse_args()

    # 1. Load the compiled model
    print(f"Loading model from {MODEL_PATH}...")
    if not MODEL_PATH.exists():
        print("Error: Model file not found. Please ensure training has completed.")
        sys.exit(1)
        
    model = tf.keras.models.load_model(str(MODEL_PATH))

    # 2. Load the target scaler for inverse transformation
    print(f"Loading scaler from {SCALER_PATH}...")
    if not SCALER_PATH.exists():
        print("Error: Scaler file not found.")
        sys.exit(1)
        
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    # 3. Process the input image
    print(f"Processing image: {args.image}")
    try:
        input_tensor = load_and_preprocess_image(args.image)
    except Exception as e:
        print(f"Failed to process image: {e}")
        sys.exit(1)

    # 4. Run Inference
    print("Running inference...\n")
    preds = model.predict(input_tensor, verbose=0)
    
    # Keras models with multiple named output heads return a dictionary
    # mapping the head name to the predicted batch array.
    scaled_preds = []
    for col in TARGET_COLS:
        if isinstance(preds, dict):
            # dict output: {'PM2.5': array([[val]]), ...}
            val = preds[col][0][0]
        else:
            # list output fallback
            idx = TARGET_COLS.index(col)
            val = preds[idx][0][0]
        scaled_preds.append(val)
        
    # 5. Inverse Transform (Convert standard scaled predictions back to real-world units)
    # Scaler expects shape (n_samples, n_features)
    scaled_preds_array = np.array(scaled_preds).reshape(1, -1)
    real_preds = scaler.inverse_transform(scaled_preds_array)[0]

    # 6. Display Results
    print("=" * 45)
    print("  AirSight Predictions (Real-world units)")
    print("=" * 45)
    print(f"  {'Target':<15} |  {'Predicted Value':>15}")
    print("-" * 45)
    
    for col, val in zip(TARGET_COLS, real_preds):
        print(f"  {col:<15} |  {val:>15.2f}")
    print("=" * 45)


if __name__ == "__main__":
    main()

"""
evaluate.py — AirSight systematic accuracy benchmark.

Runs the trained CNN over every image in data/test.csv and computes
per-pollutant MAE, RMSE, and R² in both scaled and real-world units.

IMPORTANT: Run this from ~/airsight (WSL).
The image paths in test.csv are Linux-absolute paths (/home/nytrixis/...).
They will not resolve if you run from /mnt/c/.

Usage
-----
    cd ~/airsight
    source linux_env/bin/activate
    python evaluate.py

Output
------
    • Per-pollutant metrics table printed to terminal (colour-coded by R²)
    • artifacts/evaluation_report.csv   — per-image predictions vs ground truth
    • artifacts/metrics_summary.csv     — one-row-per-pollutant metrics table
"""

import os
import sys
import pickle
import warnings
from pathlib import Path

# ── Silence verbose logs before heavy imports ─────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Repo layout ───────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent
SRC_DIR     = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

MODEL_PATH  = REPO_ROOT / "artifacts" / "best_model.keras"
SCALER_PATH = REPO_ROOT / "artifacts" / "scaler.pkl"
TEST_CSV    = REPO_ROOT / "data" / "test.csv"
REPORT_PATH = REPO_ROOT / "artifacts" / "evaluation_report.csv"
METRICS_PATH = REPO_ROOT / "artifacts" / "metrics_summary.csv"

IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32   # increase if you have more VRAM

from model import TARGET_COLS  # noqa: E402

# ── ANSI colour helpers ───────────────────────────────────────────────────────
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"
DIM    = "\033[2m"
W      = 74   # banner width

POLLUTANT_UNITS = {
    "PM2.5": "µg/m³", "PM10": "µg/m³",
    "O3":    "ppb",   "CO":   "ppm",
    "SO2":   "ppb",   "NO2":  "ppb",
    "AQI":   "—",
}


# ─────────────────────────────────────────────────────────────────────────────
# Resource loading
# ─────────────────────────────────────────────────────────────────────────────

def load_resources():
    """Load the Keras model and sklearn scaler."""
    print(f"\n{DIM}⚙  Loading model  →  {MODEL_PATH.name}{RESET}")
    if not MODEL_PATH.exists():
        print(f"❌  Model not found at {MODEL_PATH}")
        sys.exit(1)
    model = tf.keras.models.load_model(str(MODEL_PATH))

    print(f"{DIM}⚙  Loading scaler →  {SCALER_PATH.name}{RESET}")
    if not SCALER_PATH.exists():
        print(f"❌  Scaler not found at {SCALER_PATH}")
        sys.exit(1)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


# ─────────────────────────────────────────────────────────────────────────────
# tf.data inference pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess(path: tf.Tensor) -> tf.Tensor:
    """Read + decode + resize + ResNet50 normalise."""
    raw   = tf.io.read_file(path)
    image = tf.image.decode_jpeg(raw, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image


def build_inference_dataset(paths: list) -> tf.data.Dataset:
    """Build an optimised tf.data pipeline for batch inference."""
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def run_batch_inference(model, paths: list) -> np.ndarray:
    """
    Run model over all images in batches.

    Returns
    -------
    np.ndarray of shape (N, 7) in scaled space,
    columns ordered to match TARGET_COLS.
    """
    ds        = build_inference_dataset(paths)
    all_preds = {col: [] for col in TARGET_COLS}
    total     = len(paths)
    processed = 0

    for batch in ds:
        raw = model.predict(batch, verbose=0)
        for col in TARGET_COLS:
            if isinstance(raw, dict):
                all_preds[col].extend(raw[col].flatten().tolist())
            else:
                idx = TARGET_COLS.index(col)
                all_preds[col].extend(raw[idx].flatten().tolist())

        processed += int(batch.shape[0])
        pct = processed / total * 100
        bar_len   = 30
        filled    = int(bar_len * processed / total)
        bar       = "█" * filled + "░" * (bar_len - filled)
        print(
            f"\r  {DIM}[{bar}] {processed}/{total}  ({pct:.1f}%){RESET}",
            end="", flush=True,
        )

    print()   # newline after progress bar
    return np.column_stack([all_preds[col] for col in TARGET_COLS])


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _r2_colour(r2: float) -> str:
    if r2 >= 0.9: return GREEN
    if r2 >= 0.7: return YELLOW
    return RED


def _banner(title: str):
    print(f"\n{'═' * W}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{'═' * W}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    _banner("📊  AirSight  ·  Accuracy Evaluation")

    # ── Load CSV ──────────────────────────────────────────────────────────────
    if not TEST_CSV.exists():
        print(f"❌  test.csv not found at {TEST_CSV}")
        sys.exit(1)

    df    = pd.read_csv(TEST_CSV)
    paths = df["image_path"].values.tolist()

    print(f"\n  Dataset : {TEST_CSV}")
    print(f"  Samples : {len(df):,}")
    print(f"  Targets : {', '.join(TARGET_COLS)}")

    # ── Validate image paths ──────────────────────────────────────────────────
    print(f"\n{DIM}  Checking image paths ...{RESET}")
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        print(f"\n  {RED}❌  {len(missing):,} image paths not found on this filesystem.{RESET}")
        print(f"     Example: {missing[0]}")
        print(
            f"\n  {BOLD}FIX:{RESET} Run evaluate.py from ~/airsight (WSL), not /mnt/c/.\n"
            f"  The CSV stores Linux-absolute paths (/home/nytrixis/...)."
        )
        sys.exit(1)

    print(f"  {GREEN}✅  All {len(paths):,} image paths verified.{RESET}")

    # ── Load resources ────────────────────────────────────────────────────────
    model, scaler = load_resources()

    # ── Ground truth in scaled space ──────────────────────────────────────────
    # test.csv stores Z-score normalised labels, same scale the model outputs
    gt_scaled = df[TARGET_COLS].values.astype(np.float32)   # (N, 7)

    # ── Batch inference ───────────────────────────────────────────────────────
    _banner("🔭  Batch Inference")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"  Batches    : {int(np.ceil(len(paths) / BATCH_SIZE))}\n")
    pred_scaled = run_batch_inference(model, paths)           # (N, 7)

    # ── Inverse transform to real-world units ─────────────────────────────────
    pred_real = scaler.inverse_transform(pred_scaled)         # (N, 7)
    gt_real   = scaler.inverse_transform(gt_scaled)           # (N, 7)

    # ── Per-pollutant metrics ─────────────────────────────────────────────────
    _banner("📈  Per-Pollutant Metrics")
    print(
        f"  {'Pollutant':<10}  {'MAE':>8}  {'RMSE':>8}  {'R²':>7}  "
        f"{'MAE (real)':>12}  Unit"
    )
    print(
        f"  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*12}  {'─'*6}"
    )

    metric_rows = []
    for i, col in enumerate(TARGET_COLS):
        y_true_s = gt_scaled[:, i]
        y_pred_s = pred_scaled[:, i]
        y_true_r = gt_real[:, i]
        y_pred_r = pred_real[:, i]

        mae_s  = float(mean_absolute_error(y_true_s, y_pred_s))
        rmse_s = float(np.sqrt(mean_squared_error(y_true_s, y_pred_s)))
        r2     = float(r2_score(y_true_s, y_pred_s))
        mae_r  = float(mean_absolute_error(y_true_r, y_pred_r))
        unit   = POLLUTANT_UNITS.get(col, "")
        color  = _r2_colour(r2)

        print(
            f"  {col:<10}  {mae_s:>8.4f}  {rmse_s:>8.4f}  "
            f"{color}{r2:>7.4f}{RESET}  {mae_r:>12.2f}  {unit}"
        )
        metric_rows.append({
            "pollutant":   col,
            "mae_scaled":  round(mae_s,  6),
            "rmse_scaled": round(rmse_s, 6),
            "r2":          round(r2,     6),
            "mae_real":    round(mae_r,  4),
            "unit":        unit,
        })

    # Overall summary
    overall_r2 = float(np.mean([r["r2"] for r in metric_rows]))
    color = _r2_colour(overall_r2)
    print(f"{'─' * W}")
    print(
        f"  {BOLD}Overall mean R²{RESET}  :  "
        f"{color}{BOLD}{overall_r2:.4f}{RESET}"
    )
    print(f"{'═' * W}")

    # Flag weak pollutants
    weak = [r["pollutant"] for r in metric_rows if r["r2"] < 0.5]
    if weak:
        print(
            f"\n  {RED}⚠  R² < 0.5 (needs attention): "
            f"{', '.join(weak)}{RESET}"
        )
    else:
        print(f"\n  {GREEN}✅  All pollutants have R² ≥ 0.5{RESET}")

    # ── Save per-row report ───────────────────────────────────────────────────
    report_rows = []
    for idx in range(len(df)):
        row = {"image": Path(paths[idx]).name}
        for j, col in enumerate(TARGET_COLS):
            row[f"{col}_gt"]    = round(float(gt_real[idx, j]), 4)
            row[f"{col}_pred"]  = round(float(pred_real[idx, j]), 4)
            row[f"{col}_error"] = round(float(abs(pred_real[idx, j] - gt_real[idx, j])), 4)
        report_rows.append(row)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report_rows).to_csv(REPORT_PATH, index=False)
    pd.DataFrame(metric_rows).to_csv(METRICS_PATH, index=False)

    print(f"\n  💾  Per-row report  →  {REPORT_PATH}")
    print(f"  💾  Metrics summary →  {METRICS_PATH}\n")


if __name__ == "__main__":
    main()

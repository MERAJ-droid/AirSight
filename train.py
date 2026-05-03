"""
train.py — AirSight two-phase training script.

Usage (from repo root, with venv active):
    python train.py

Phase 1 : Frozen ResNet50 backbone, train 7 regression heads  (lr=1e-3, 10 epochs max)
Phase 2 : Unfreeze backbone, full fine-tune                   (lr=1e-5, 15 epochs max)
"""

import sys
import os
from pathlib import Path

# ── sys.path: make src/ importable when run from any CWD ─────────────────────
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import tensorflow as tf
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
from dataset import make_dataset
from model import build_model, TARGET_COLS

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR      = REPO_ROOT / "data"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = str(ARTIFACTS_DIR / "best_model.keras")

# ── Config ────────────────────────────────────────────────────────────────────
BATCH_SIZE    = 16
PHASE1_EPOCHS = 10
PHASE2_EPOCHS = 15
PHASE1_LR     = 1e-3
PHASE2_LR     = 1e-5


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:

    # ── GPU check ─────────────────────────────────────────────────────────────
    gpus = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow : {tf.__version__}")
    print(f"GPUs found : {[g.name for g in gpus] if gpus else 'NONE — running on CPU'}")
    print(f"Repo root  : {REPO_ROOT}")
    print(f"Targets    : {TARGET_COLS}\n")

    # ── Datasets ──────────────────────────────────────────────────────────────
    print("Building tf.data pipelines...")
    train_ds = make_dataset(DATA_DIR / "train.csv", batch_size=BATCH_SIZE, training=True)
    val_ds   = make_dataset(DATA_DIR / "val.csv",   batch_size=BATCH_SIZE, training=False)
    test_ds  = make_dataset(DATA_DIR / "test.csv",  batch_size=BATCH_SIZE, training=False)

    print(f"  Train batches : {len(train_ds)}")
    print(f"  Val   batches : {len(val_ds)}")
    print(f"  Test  batches : {len(test_ds)}\n")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("Building model...")
    model = build_model(learning_rate=PHASE1_LR)
    model.summary(line_length=80, show_trainable=True)

    # ── Callbacks (shared across both phases) ─────────────────────────────────
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=CHECKPOINT_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # ── Phase 1: Frozen backbone ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"PHASE 1 — Frozen backbone  |  lr={PHASE1_LR}  |  max {PHASE1_EPOCHS} epochs")
    print("=" * 60)

    history_p1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE1_EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    print(f"\nPhase 1 complete.")
    print(f"  Epochs run    : {len(history_p1.epoch)}")
    print(f"  Best val_loss : {min(history_p1.history['val_loss']):.4f}")

    # ── Phase 2: Unfreeze & fine-tune ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"PHASE 2 — Full fine-tune  |  lr={PHASE2_LR}  |  max {PHASE2_EPOCHS} epochs")
    print("=" * 60)

    # Unfreeze backbone
    backbone = model.get_layer("resnet50")
    backbone.trainable = True

    # MANDATORY recompile after changing trainable flags
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=PHASE2_LR),
        loss={col: "mse" for col in TARGET_COLS},
        metrics={col: ["mae"] for col in TARGET_COLS},
    )

    trainable_params = sum(tf.size(v).numpy() for v in model.trainable_variables)
    print(f"Trainable params after unfreeze : {trainable_params:,}\n")

    history_p2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE2_EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    print(f"\nPhase 2 complete.")
    print(f"  Epochs run    : {len(history_p2.epoch)}")
    print(f"  Best val_loss : {min(history_p2.history['val_loss']):.4f}")

    # ── Final evaluation ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL EVALUATION on test set")
    print("=" * 60)

    results = model.evaluate(test_ds, verbose=1, return_dict=True)

    print("\n── Per-pollutant Test Metrics (scaled units) ──────────────")
    print(f"  {'Pollutant':<10}  {'MAE':>8}   {'MSE':>10}")
    print(f"  {'-'*10}  {'-'*8}   {'-'*10}")
    for col in TARGET_COLS:
        mae = results.get(f"{col}_mae", float("nan"))
        mse = results.get(f"{col}_loss", float("nan"))
        print(f"  {col:<10}  {mae:>8.4f}   {mse:>10.4f}")

    print(f"\n  Aggregate loss (sum of MSEs) : {results['loss']:.4f}")
    print(f"\nBest model checkpoint : {CHECKPOINT_PATH}")
    print("\nTraining complete.")


if __name__ == "__main__":
    main()

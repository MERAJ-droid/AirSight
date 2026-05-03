"""
model.py — ResNet50-based multi-head regression model for AirSight.

Public API
----------
build_model(learning_rate=1e-3) -> tf.keras.Model
    Returns a compiled Keras model with 7 separate Dense(1) output heads,
    one per pollutant: PM2.5, PM10, O3, CO, SO2, NO2, AQI.
"""

import tensorflow as tf

TARGET_COLS = ["PM2.5", "PM10", "O3", "CO", "SO2", "NO2", "AQI"]


def build_model(learning_rate: float = 1e-3) -> tf.keras.Model:
    """
    Build and compile a frozen ResNet50 backbone with 7 regression heads.

    Architecture
    ------------
    Input (224, 224, 3)
        └─ ResNet50 (imagenet weights, include_top=False)  [FROZEN]
            └─ GlobalAveragePooling2D
                ├─ Dense(256, relu) -> Dense(1)  ->  PM2.5
                ├─ Dense(256, relu) -> Dense(1)  ->  PM10
                ├─ Dense(256, relu) -> Dense(1)  ->  O3
                ├─ Dense(256, relu) -> Dense(1)  ->  CO
                ├─ Dense(256, relu) -> Dense(1)  ->  SO2
                ├─ Dense(256, relu) -> Dense(1)  ->  NO2
                └─ Dense(256, relu) -> Dense(1)  ->  AQI

    Compilation
    -----------
    Optimizer : Adam(learning_rate)
    Loss      : mse  (one per head)
    Metric    : mae  (one per head)

    Parameters
    ----------
    learning_rate : Initial LR for the Adam optimiser (default 1e-3).

    Returns
    -------
    Compiled tf.keras.Model.
    """
    # ── Backbone ─────────────────────────────────────────────────────────────
    backbone = tf.keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )
    backbone.trainable = False   # freeze all backbone weights

    # ── Shared feature stem ───────────────────────────────────────────────────
    inputs = tf.keras.Input(shape=(224, 224, 3), name="image")
    x      = backbone(inputs, training=False)   # BN layers stay in inference mode
    shared = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)

    # ── Per-pollutant regression heads ────────────────────────────────────────
    outputs = {}
    for col in TARGET_COLS:
        # Sanitise column name for use as a Keras layer name (PM2.5 -> PM2_5)
        safe_name = col.replace(".", "_")
        h = tf.keras.layers.Dense(256, activation="relu", name=f"{safe_name}_dense")(shared)
        outputs[col] = tf.keras.layers.Dense(1, name=col)(h)

    # ── Build model ───────────────────────────────────────────────────────────
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="airsight_resnet50")

    # ── Compile ───────────────────────────────────────────────────────────────
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={col: "mse" for col in TARGET_COLS},
        metrics={col: ["mae"] for col in TARGET_COLS},
    )

    return model

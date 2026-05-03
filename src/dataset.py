"""
dataset.py — tf.data pipeline for AirSight regression.

Public API
----------
make_dataset(csv_path, batch_size=32, training=False) -> tf.data.Dataset
    Yields (image_tensor, label_dict) batches ready for model.fit().
"""

import tensorflow as tf
import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE    = (224, 224)
TARGET_COLS = ["PM2.5", "PM10", "O3", "CO", "SO2", "NO2", "AQI"]


def _load_and_preprocess(image_path: tf.Tensor) -> tf.Tensor:
    """Read a JPEG from disk, decode, resize, and apply ResNet50 preprocessing."""
    raw   = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(raw, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image


def _augment(image: tf.Tensor) -> tf.Tensor:
    """Light augmentation applied only during training."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.15)
    return image


def make_dataset(
    csv_path: str,
    batch_size: int = 32,
    training: bool = False,
) -> tf.data.Dataset:
    """
    Build a tf.data.Dataset from a split CSV produced by 02_build_dataset.ipynb.

    Parameters
    ----------
    csv_path   : Path to train.csv / val.csv / test.csv.
    batch_size : Mini-batch size (default 32).
    training   : If True, apply augmentation and shuffle; False for val/test.

    Returns
    -------
    tf.data.Dataset yielding (image, label_dict) batches.
        image      : float32 tensor of shape (B, 224, 224, 3), ResNet50-scaled.
        label_dict : dict mapping each pollutant name -> float32 tensor (B, 1).
    """
    df = pd.read_csv(csv_path)

    # ── Build parallel lists ─────────────────────────────────────────────────
    paths  = df["image_path"].values.tolist()
    labels = {col: df[col].values.astype("float32") for col in TARGET_COLS}

    # ── Construct dataset ────────────────────────────────────────────────────
    path_ds  = tf.data.Dataset.from_tensor_slices(paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    ds       = tf.data.Dataset.zip((path_ds, label_ds))

    if training:
        ds = ds.shuffle(buffer_size=len(df), reshuffle_each_iteration=True)

    # Map: load image (parallelised), optionally augment
    def _process(path, label):
        image = _load_and_preprocess(path)
        if training:
            image = _augment(image)
        return image, label

    ds = ds.map(_process, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=training)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

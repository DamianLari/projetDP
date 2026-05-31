"""
binary_model.py : Shared module for the Photo vs Painting binary model.

Transfer learning (MobileNetV2 or EfficientNetB0) in two phases:
  Phase 1: frozen base, we train only the head.
  Phase 2: fine-tuning of the backbone's top layers.

This module contains NO main block: it is imported by
  - 03_model_binary_photo_vs_painting.py  (training from config_binary.json)
  - tune_binary.py                         (Optuna hyperparameter search)

Everything is driven by a `cfg` dict (see config_binary.json for the schema).

Key point corrected vs older version:
  Augmentation is performed BEFORE preprocess_input, in the [0, 255] space.
  (The old code augmented after preprocessing, which corrupted the input
   expected by the pre-trained backbone.)
"""
from __future__ import annotations

from datetime import datetime
import sys
import time
from pathlib import Path

# Allows importing utils from the root of the Deliverable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tensorflow as tf
from tensorflow.keras import callbacks, layers, models

from utils import SEED
SPLIT_DIR = Path(__file__).resolve().parent.parent / "data_split"

BINARY_CLASSES = ["Painting", "Photo"]
AUTOTUNE = tf.data.AUTOTUNE

# Available backbones: (constructor, preprocessing function)

def _get_backbone(cfg: dict, img_h: int, img_w: int):
    """Returns (unbuilt_base_model_factory, preprocess_fn) based on cfg["backbone"]."""
    name = cfg.get("backbone", "MobileNetV2")
    common = dict(include_top=False, weights="imagenet",
                  input_shape=(img_h, img_w, 3), pooling=None)

    if name == "MobileNetV2":
        base = tf.keras.applications.MobileNetV2(**common)
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    elif name == "EfficientNetB0":
        base = tf.keras.applications.EfficientNetB0(**common)
        preprocess = tf.keras.applications.efficientnet.preprocess_input
    else:
        raise ValueError(f"Unknown backbone: {name!r} (expected MobileNetV2 or EfficientNetB0)")

    return base, preprocess

# Custom tf.data pipeline (same logic as tune_023.py, for consistency)

def _decode_img(file_path: tf.Tensor, img_h: int, img_w: int) -> tf.Tensor:
    """Decodes + resizes. Returns uint8 [0,255] (compact for caching)."""
    raw = tf.io.read_file(file_path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, [img_h, img_w])
    return tf.cast(img, tf.uint8)


def _get_label(file_path: tf.Tensor) -> tf.Tensor:
    """Binary float label: Photo=1.0, Painting=0.0 (based on BINARY_CLASSES)."""
    parts = tf.strings.split(file_path, "/")
    n = tf.shape(parts)[0]
    class_name = parts[n - 2]
    matches = tf.cast(tf.equal(class_name, tf.constant(BINARY_CLASSES)), tf.int32)
    return tf.cast(tf.argmax(matches), tf.float32)


def _make_augment_fn(cfg: dict, img_h: int, img_w: int):
    """Augmentation inside the [0, 255] space (BEFORE preprocess_input)."""
    aug = cfg.get("augmentation", {})
    brightness = aug.get("brightness", 0.1)
    contrast   = aug.get("contrast", 0.1)
    saturation = aug.get("saturation", 0.1)
    zoom       = aug.get("zoom", 0.1)

    def _augment(img: tf.Tensor) -> tf.Tensor:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=brightness * 255.0)
        img = tf.image.random_contrast(img, lower=1 - contrast, upper=1 + contrast)
        img = tf.image.random_saturation(img, lower=1 - saturation, upper=1 + saturation)
        shape = tf.shape(img)
        crop_size = tf.cast(tf.cast(shape[:2], tf.float32) * (1 - zoom), tf.int32)
        img = tf.image.random_crop(img, size=[crop_size[0], crop_size[1], 3])
        img = tf.image.resize(img, [img_h, img_w])
        return tf.clip_by_value(img, 0.0, 255.0)

    return _augment


def make_dataset(subset: str, cfg: dict, augment: bool = False) -> tf.data.Dataset:
    """Optimized tf.data pipeline."""
    img_h, img_w = cfg["img_size"]
    batch_size = cfg["batch_size"]
    preprocess = _preprocess_for(cfg)

    # File listing for the 2 classes only (Painting / Photo)
    patterns = [str(SPLIT_DIR / subset / c / "*") for c in BINARY_CLASSES]
    ds = tf.data.Dataset.list_files(patterns, shuffle=False, seed=cfg["seed"])

    # Parallelized decoding + resizing across all CPU cores
    ds = ds.map(lambda p: (_decode_img(p, img_h, img_w), _get_label(p)),
                num_parallel_calls=AUTOTUNE)

    # --- cache: expensive CPU workloads (decoding+resizing) are only done once ---
    cache_dir = cfg.get("cache_dir")
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        ds = ds.cache(str(Path(cache_dir) / f"{subset}_{img_h}x{img_w}.cache"))
    else:
        ds = ds.cache()  # in RAM

    # Shuffle after cache (cheap operation on uint8 elements)
    if augment:
        ds = ds.shuffle(cfg.get("shuffle_buffer", 1000), seed=cfg["seed"],
                        reshuffle_each_iteration=True)

    # Float32 casting + augmentation (train) + backbone preprocessing LAST
    ds = ds.map(lambda img, lbl: (tf.cast(img, tf.float32), lbl),
                num_parallel_calls=AUTOTUNE)
    if augment:
        aug_fn = _make_augment_fn(cfg, img_h, img_w)
        ds = ds.map(lambda img, lbl: (aug_fn(img), lbl), num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda img, lbl: (preprocess(img), lbl), num_parallel_calls=AUTOTUNE)

    return ds.batch(batch_size).prefetch(AUTOTUNE)


def _preprocess_for(cfg: dict):
    name = cfg.get("backbone", "MobileNetV2")
    if name == "MobileNetV2":
        return tf.keras.applications.mobilenet_v2.preprocess_input
    if name == "EfficientNetB0":
        return tf.keras.applications.efficientnet.preprocess_input
    raise ValueError(f"Unknown backbone: {name!r}")

# Model

def build_model(cfg: dict) -> models.Model:
    """Builds the model (frozen base). Unfreezing happens in run_training (phase 2)."""
    img_h, img_w = cfg["img_size"]
    base, _ = _get_backbone(cfg, img_h, img_w)
    base.trainable = False

    inp = layers.Input(shape=(img_h, img_w, 3), name="input_image")
    # training=False: backbone's BatchNorm remains in inference mode (Keras
    # recommended recipe for fine-tuning a pre-trained network).
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(cfg["dense_units"], activation="relu")(x)
    x = layers.Dropout(cfg["dropout_head"])(x)
    out = layers.Dense(1, activation="sigmoid", name="predictions")(x)

    model = models.Model(inp, out, name=f"{cfg.get('backbone', 'MobileNetV2')}_binary")
    model._backbone_name = base.name  # used to target the base network during fine-tuning
    return model


def _unfreeze_base(model: models.Model, cfg: dict) -> int:
    """Unfreezes the last layers of the backbone. Returns the number of trainable params."""
    base = model.get_layer(model._backbone_name)
    base.trainable = True
    frac = cfg.get("unfreeze_fraction", 0.3)
    cut = int(len(base.layers) * (1 - frac))
    for layer in base.layers[:cut]:
        layer.trainable = False
    # BatchNorm layers remain frozen even within the unfrozen block (best practice)
    for layer in base.layers[cut:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    return sum(int(tf.size(v).numpy()) for v in model.trainable_variables)

# Two-phase Training (reused by both scripts)

def run_training(
    cfg: dict,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    model_path: str | Path | None = None,
    extra_callbacks: list | None = None,
    verbose: int = 1,
) -> tuple[models.Model, dict]:
    """Trains the binary model in two phases and returns (model, merged_history).

    `extra_callbacks` is appended to phase 2 (useful for the tuner: SpeedGuard...).
    `model_path`: if provided, ModelCheckpoint saves the best model.
    """
    metrics = ["accuracy", tf.keras.metrics.AUC(name="auc")]

    # Phase 1: head only
    model = build_model(cfg)
    if verbose:
        model.summary(print_fn=print)
        n_head = sum(int(tf.size(v).numpy()) for v in model.trainable_variables)
        print(f"\nTrainable parameters (phase 1, head): {n_head:,}")

    model.compile(optimizer=tf.keras.optimizers.Adam(cfg["lr_head"]),
                  loss="binary_crossentropy", metrics=metrics)

    cb1 = [callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                   restore_best_weights=True, verbose=verbose)]
    if model_path:
        cb1.append(callbacks.ModelCheckpoint(str(model_path), monitor="val_loss",
                                             save_best_only=True, verbose=verbose))

    t0 = time.time()

    log_dir_phase1 = (Path("logs") / "binary" / "phase1" / datetime.now().strftime("%Y%m%d-%H%M%S"))

    cb1.append(callbacks.TensorBoard(log_dir=str(log_dir_phase1), histogram_freq=1))

    hist1 = model.fit(train_ds, validation_data=val_ds, epochs=cfg["epochs_head"], callbacks=cb1, verbose=verbose)
    if verbose:
        print(f"Phase 1: {(time.time()-t0)/60:.1f} min")

    # Phase 2: fine-tuning
    n_ft = _unfreeze_base(model, cfg)
    if verbose:
        print(f"Trainable parameters (phase 2, fine-tuning): {n_ft:,}")

    model.compile(optimizer=tf.keras.optimizers.Adam(cfg["lr_finetune"]),
                  loss="binary_crossentropy", metrics=metrics)

    cb2 = [
        callbacks.EarlyStopping(monitor="val_loss", patience=cfg["early_stopping_patience"],
                                restore_best_weights=True, verbose=verbose),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=cfg["reduce_lr_factor"],
                                    patience=cfg["reduce_lr_patience"], min_lr=1e-7,
                                    verbose=verbose),
    ]
    if model_path:
        cb2.append(callbacks.ModelCheckpoint(str(model_path), monitor="val_loss", save_best_only=True, verbose=verbose))
    if extra_callbacks:
        cb2.extend(extra_callbacks)

    t0 = time.time()

    log_dir_phase2 = (Path("logs") / "binary" / "phase2" / datetime.now().strftime("%Y%m%d-%H%M%S"))

    cb2.append(callbacks.TensorBoard(log_dir=str(log_dir_phase2), histogram_freq=1))

    hist2 = model.fit(train_ds, validation_data=val_ds, epochs=cfg["epochs_finetune"], callbacks=cb2, verbose=verbose)
    if verbose:
        print(f"Phase 2: {(time.time()-t0)/60:.1f} min")

    # Merge the two histories
    full_history = {k: list(hist1.history.get(k, [])) + list(hist2.history.get(k, []))
                    for k in set(hist1.history) | set(hist2.history)}
    return model, full_history

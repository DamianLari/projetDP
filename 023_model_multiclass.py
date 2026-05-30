"""
023 — CNN multi-class 5 classes (pipeline tf.data custom).

Différence vs 022 : pipeline de données entièrement manuel avec tf.data.
- Lecture fichiers par fichiers avec map + num_parallel_calls=AUTOTUNE
- Décodage JPEG/PNG parallélisé sur tous les CPU cores disponibles
- Augmentation déplacée dans le pipeline tf.data (hors modèle) → sur CPU en parallèle
  pendant que le GPU calcule le batch précédent
- interleave pour maximiser le débit disque
- Pas de cache VRAM → évite les OOM GPU

Sortie :
    - models/multiclass_023_best.keras
    - histories/multiclass_023_history.json
    - figures/023_multiclass_curves.png

Usage :
    python3 023_model_multiclass.py
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models, regularizers

from utils import (
    BATCH_SIZE, CFG, CLASS_NAMES, HISTORIES_DIR, IMG_SIZE, MODELS_DIR, SEED, SPLIT_DIR,
    save_history, set_seeds,
)


FIGURES_DIR = Path("figures")
EPOCHS = CFG["epochs"]
IMG_H, IMG_W = IMG_SIZE
AUTOTUNE = tf.data.AUTOTUNE
N_CLASSES = len(CLASS_NAMES)


# ----------------------------------------------------------------------------
# Pipeline tf.data custom
# ----------------------------------------------------------------------------

def _get_label(file_path: tf.Tensor) -> tf.Tensor:
    parts = tf.strings.split(file_path, "/")
    class_name = parts[-2]
    return tf.argmax(tf.cast(tf.equal(class_name, CLASS_NAMES), tf.int32))


def _decode_img(file_path: tf.Tensor) -> tf.Tensor:
    raw = tf.io.read_file(file_path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_H, IMG_W])
    return tf.cast(img, tf.float32) / 255.0


def _augment(img: tf.Tensor) -> tf.Tensor:
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    img = tf.image.random_saturation(img, lower=0.9, upper=1.1)
    # crop aléatoire (simule zoom/translation) : crop 90% puis resize
    shape = tf.shape(img)
    crop_size = tf.cast(tf.cast(shape[:2], tf.float32) * 0.9, tf.int32)
    img = tf.image.random_crop(img, size=[crop_size[0], crop_size[1], 3])
    img = tf.image.resize(img, [IMG_H, IMG_W])
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img


def make_dataset(subset: str, augment: bool = False) -> tf.data.Dataset:
    pattern = str(SPLIT_DIR / subset / "*" / "*")
    ds = tf.data.Dataset.list_files(pattern, shuffle=(subset == "train"), seed=SEED)

    ds = ds.map(
        lambda p: (_decode_img(p), _get_label(p)),
        num_parallel_calls=AUTOTUNE,
    )

    if augment:
        ds = ds.map(
            lambda img, lbl: (_augment(img), lbl),
            num_parallel_calls=AUTOTUNE,
        )

    if subset == "train":
        ds = ds.shuffle(CFG["shuffle_buffer"], seed=SEED, reshuffle_each_iteration=True)

    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTOTUNE)
    return ds


# ----------------------------------------------------------------------------
# Modèle
# ----------------------------------------------------------------------------

def build_model(n_classes: int = N_CLASSES,
                img_size: tuple[int, int] = IMG_SIZE) -> models.Model:
    dp = CFG["dropout"]
    filters = CFG["filters"]
    inp = layers.Input(shape=(img_size[1], img_size[0], 3), name="input_image")
    x = inp

    dropouts = [dp["bloc1"], dp["bloc2"], dp["bloc3"], dp["bloc4"]]
    for i, (f, d) in enumerate(zip(filters, dropouts)):
        x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        if i < 2:
            x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
            x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(d)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(CFG["dense_units"], activation="relu",
                     kernel_regularizer=regularizers.l2(CFG["l2"]))(x)
    x = layers.Dropout(dp["head"])(x)
    out = layers.Dense(n_classes, activation="softmax", name="predictions")(x)

    return models.Model(inp, out, name="cnn_multiclass_023")


def plot_curves(history_dict: dict, save_path: Path) -> None:
    epochs = range(1, len(history_dict["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(epochs, history_dict["loss"], label="train loss")
    axes[0].plot(epochs, history_dict["val_loss"], label="val loss")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, history_dict["accuracy"], label="train acc")
    axes[1].plot(epochs, history_dict["val_accuracy"], label="val acc")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def main() -> None:
    set_seeds(SEED)
    MODELS_DIR.mkdir(exist_ok=True)
    HISTORIES_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print(f"TF version : {tf.__version__}")
    print(f"GPUs       : {tf.config.list_physical_devices('GPU')}")
    print(f"IMG_SIZE   : {IMG_SIZE}  BATCH : {BATCH_SIZE}  EPOCHS : {EPOCHS}")
    print("=" * 70)

    if not SPLIT_DIR.exists():
        raise SystemExit(f"ERREUR : {SPLIT_DIR} introuvable. Lance d'abord 01_data_split_analysis.py.")

    print("\n1. Construction des pipelines tf.data...")
    train_ds = make_dataset("train", augment=True)
    val_ds   = make_dataset("val",   augment=False)
    test_ds  = make_dataset("test",  augment=False)
    print("Pipelines prêts.")

    print("\n2. Construction du modèle...")
    model = build_model()
    model.summary(print_fn=print)
    print(f"\nTotal params : {model.count_params():,}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=CFG["learning_rate"]),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    model_path = MODELS_DIR / "multiclass_023_best.keras"
    history_path = HISTORIES_DIR / "multiclass_023_history.json"
    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=CFG["early_stopping_patience"],
                                restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=CFG["reduce_lr_factor"],
                                    patience=CFG["reduce_lr_patience"], min_lr=1e-6, verbose=1),
        callbacks.ModelCheckpoint(str(model_path), monitor="val_loss", save_best_only=True, verbose=1),
    ]

    print(f"\n3. Entraînement (max {EPOCHS} epochs)...")
    t_start = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=cb, verbose=1)
    train_time = time.time() - t_start
    print(f"\nEntraînement : {train_time / 60:.1f} min — {len(history.history['loss'])} epochs")

    save_history(history, history_path)
    print(f"History : {history_path}")
    print(f"Modèle  : {model_path}")

    fig_path = FIGURES_DIR / "023_multiclass_curves.png"
    plot_curves(history.history, fig_path)
    print(f"Courbes : {fig_path}")

    print("\n4. Évaluation sur le test set...")
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"Test loss     : {test_loss:.4f}")
    print(f"Test accuracy : {test_acc:.4f}")

    print("\n" + "=" * 70)
    print(f"Terminé. Modèle : {model_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

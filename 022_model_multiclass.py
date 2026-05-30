"""
022 — CNN multi-class 5 classes (architecture 02, chargement à la volée).

Identique à 02 en termes d'architecture (4 blocs Conv-BN-Conv-BN-MaxPool-Dropout
+ GlobalAveragePooling2D), mais charge les images via image_dataset_from_directory
(pas de chargement en RAM) pour éviter les OOM sur gros datasets.

Sortie :
    - models/multiclass_022_best.keras
    - histories/multiclass_022_history.json
    - figures/022_multiclass_curves.png

Usage :
    python3 022_model_multiclass.py
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
    make_augmentation_layer, save_history, set_seeds,
)


FIGURES_DIR = Path("figures")
EPOCHS = CFG["epochs"]
IMG_H, IMG_W = IMG_SIZE
AUTOTUNE = tf.data.AUTOTUNE


def build_multiclass_model(n_classes: int = 5,
                            img_size: tuple[int, int] = IMG_SIZE,
                            augment: tf.keras.Sequential | None = None) -> models.Model:
    dp = CFG["dropout"]
    filters = CFG["filters"]
    inp = layers.Input(shape=(img_size[1], img_size[0], 3), name="input_image")

    x = augment(inp) if augment is not None else inp
    x = layers.Rescaling(1.0 / 255)(x)

    # Blocs Conv dynamiques (selon filters dans config.json)
    dropouts = [dp["bloc1"], dp["bloc2"], dp["bloc3"], dp["bloc4"]]
    for i, (f, d) in enumerate(zip(filters, dropouts)):
        x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        if i < 2:  # double conv sur les deux premiers blocs
            x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
            x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(d)(x)

    # Tête
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(CFG["dense_units"], activation="relu",
                     kernel_regularizer=regularizers.l2(CFG["l2"]))(x)
    x = layers.Dropout(dp["head"])(x)
    out = layers.Dense(n_classes, activation="softmax", name="predictions")(x)

    return models.Model(inp, out, name="cnn_multiclass_022")


def plot_curves(history_dict: dict, save_path: Path, title: str = "Multi-class") -> None:
    epochs = range(1, len(history_dict["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(epochs, history_dict["loss"], label="train loss")
    axes[0].plot(epochs, history_dict["val_loss"], label="val loss")
    axes[0].set_title(f"Loss — {title}"); axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, history_dict["accuracy"], label="train acc")
    axes[1].plot(epochs, history_dict["val_accuracy"], label="val acc")
    axes[1].set_title(f"Accuracy — {title}"); axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
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

    # --- Chargement à la volée (pas de RAM) ---
    print("\n1. Chargement datasets depuis data_split/...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        SPLIT_DIR / "train",
        class_names=CLASS_NAMES,
        seed=SEED,
        image_size=(IMG_H, IMG_W),
        batch_size=BATCH_SIZE,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        SPLIT_DIR / "val",
        class_names=CLASS_NAMES,
        seed=SEED,
        image_size=(IMG_H, IMG_W),
        batch_size=BATCH_SIZE,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        SPLIT_DIR / "test",
        class_names=CLASS_NAMES,
        seed=SEED,
        image_size=(IMG_H, IMG_W),
        batch_size=BATCH_SIZE,
    )
    n_classes = len(CLASS_NAMES)
    print(f"Classes : {train_ds.class_names}")

    train_ds = train_ds.cache().shuffle(CFG["shuffle_buffer"], seed=SEED).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    test_ds = test_ds.cache().prefetch(AUTOTUNE)

    # --- Modèle ---
    print("\n2. Construction du modèle (avec augmentation intégrée)...")
    augment = make_augmentation_layer()
    model = build_multiclass_model(n_classes=n_classes, img_size=IMG_SIZE, augment=augment)
    model.summary(print_fn=print)
    print(f"\nTotal params : {model.count_params():,}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=CFG["learning_rate"]),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    # --- Callbacks ---
    model_path = MODELS_DIR / "multiclass_022_best.keras"
    history_path = HISTORIES_DIR / "multiclass_022_history.json"
    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=CFG["early_stopping_patience"],
                                restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=CFG["reduce_lr_factor"],
                                    patience=CFG["reduce_lr_patience"], min_lr=1e-6, verbose=1),
        callbacks.ModelCheckpoint(str(model_path), monitor="val_loss", save_best_only=True, verbose=1),
    ]

    # --- Entraînement ---
    print(f"\n3. Entraînement (max {EPOCHS} epochs, EarlyStopping patience=5)...")
    t_start = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=cb, verbose=1)
    train_time = time.time() - t_start
    print(f"\nEntraînement : {train_time / 60:.1f} min ({train_time:.1f} s)")
    print(f"Epochs effectuées : {len(history.history['loss'])}")

    save_history(history, history_path)
    print(f"History : {history_path}")
    print(f"Modèle  : {model_path}")

    # --- Plots ---
    fig_path = FIGURES_DIR / "022_multiclass_curves.png"
    plot_curves(history.history, fig_path, title="Multi-class 022")
    print(f"Courbes : {fig_path}")

    # --- Éval ---
    print("\n4. Évaluation sur le test set...")
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"Test loss     : {test_loss:.4f}")
    print(f"Test accuracy : {test_acc:.4f}")

    print("\n" + "=" * 70)
    print(f"Terminé. Modèle : {model_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

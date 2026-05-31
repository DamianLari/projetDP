"""
022 — CNN multi-class 5 classes (style workshop CESI).

Inspiré du workshop "Réseaux de Neurones Convolutifs" (Nassim HADDAM, CESI 2020),
adapté au projet TouNum.

Caractéristiques de ce style :
- Sequential simple
- image_dataset_from_directory avec validation_split (lit directement depuis Dataset/)
- 3 blocs Conv-MaxPool (16 -> 32 -> 64 filtres) puis Flatten + Dense(128) + Dense(5)
- SparseCategoricalCrossentropy avec from_logits=True (pas de softmax explicite)
- Data augmentation + Dropout pour régularisation

Sortie :
    - models/multiclass_workshop.keras
    - histories/multiclass_workshop_history.json
    - figures/022_workshop_curves.png

Usage :
    python3 022_model_multiclass.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from utils import BATCH_SIZE, HISTORIES_DIR, IMG_SIZE, MODELS_DIR, SEED, SPLIT_DIR


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

IMG_H, IMG_W = IMG_SIZE
EPOCHS = 15          # le workshop fait 10, on ajoute un peu de marge + EarlyStopping

CLASS_NAMES = ["Painting", "Photo", "Schematics", "Sketch", "Text"]
NUM_CLASSES = len(CLASS_NAMES)
FIGURES_DIR = Path("figures")


def main() -> None:
    for d in [MODELS_DIR, HISTORIES_DIR, FIGURES_DIR]:
        d.mkdir(exist_ok=True)
    tf.keras.utils.set_random_seed(SEED)

    print("=" * 70)
    print(f"TF version : {tf.__version__}")
    print(f"GPUs       : {tf.config.list_physical_devices('GPU')}")
    print(f"Classes    : {CLASS_NAMES}")
    print(f"IMG_SIZE   : {IMG_W}x{IMG_H}  BATCH : {BATCH_SIZE}  EPOCHS max : {EPOCHS}")
    print(f"Dataset    : {SPLIT_DIR.resolve()}")
    print("=" * 70)
    if not SPLIT_DIR.exists():
        raise SystemExit(f"ERREUR : {SPLIT_DIR} introuvable. Lance d'abord 01_data_split_analysis.py.")

    # ----------------------------------------------------------------------
    # 1. Chargement avec image_dataset_from_directory (depuis data_split/)
    # ----------------------------------------------------------------------
    print("\n1. Chargement train/val depuis data_split/...")
    train_set = tf.keras.utils.image_dataset_from_directory(
        SPLIT_DIR / "train",
        class_names=CLASS_NAMES,
        seed=SEED,
        image_size=(IMG_H, IMG_W),
        batch_size=BATCH_SIZE,
    )
    val_set = tf.keras.utils.image_dataset_from_directory(
        SPLIT_DIR / "val",
        class_names=CLASS_NAMES,
        seed=SEED,
        image_size=(IMG_H, IMG_W),
        batch_size=BATCH_SIZE,
    )
    print(f"Classes utilisées : {train_set.class_names}")

    # ----------------------------------------------------------------------
    # 2. Optimisations tf.data : cache + prefetch
    # ----------------------------------------------------------------------
    AUTOTUNE = tf.data.AUTOTUNE
    train_set = train_set.cache().shuffle(1000, seed=SEED).prefetch(buffer_size=AUTOTUNE)
    val_set = val_set.cache().prefetch(buffer_size=AUTOTUNE)

    # ----------------------------------------------------------------------
    # 3. Couche de data augmentation
    # ----------------------------------------------------------------------
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),   # ±18° (0.05 × 360°)
        layers.RandomZoom(0.1),        # zoom ±10%
    ], name="data_augmentation")

    # ----------------------------------------------------------------------
    # 4. Modèle (style workshop : Sequential, 3 blocs Conv-MaxPool, Dropout, Flatten, Dense)
    # ----------------------------------------------------------------------
    print("\n2. Construction du modèle...")
    model = Sequential([
        layers.Input(shape=(IMG_H, IMG_W, 3)),
        data_augmentation,
        layers.Rescaling(1.0 / 255),

        # Bloc 1
        layers.Conv2D(16, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D(),

        # Bloc 2
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D(),

        # Bloc 3
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D(),

        # Régularisation par dropout (juste avant Flatten, comme dans le workshop)
        layers.Dropout(0.2),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(NUM_CLASSES),   # pas d'activation : logits bruts (from_logits=True dans la loss)
    ], name="cnn_workshop_multiclass")

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.summary(print_fn=print)
    print(f"\nTotal params : {model.count_params():,}")

    # ----------------------------------------------------------------------
    # 5. Callbacks
    # ----------------------------------------------------------------------
    model_path = MODELS_DIR / "multiclass_workshop.keras"
    history_path = HISTORIES_DIR / "multiclass_workshop_history.json"
    cb = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            str(model_path), monitor="val_loss", save_best_only=True, verbose=1,
        ),
    ]

    # ----------------------------------------------------------------------
    # 6. Entraînement
    # ----------------------------------------------------------------------
    print(f"\n3. Entraînement (max {EPOCHS} epochs, EarlyStopping patience=5)...")
    t0 = time.time()
    history = model.fit(
        train_set,
        validation_data=val_set,
        epochs=EPOCHS,
        callbacks=cb,
        verbose=1,
    )
    train_time = time.time() - t0
    print(f"\nEntraînement : {train_time / 60:.1f} min — {len(history.history['loss'])} epochs")

    # ----------------------------------------------------------------------
    # 7. Sauvegarde history JSON
    # ----------------------------------------------------------------------
    hist_data = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(history_path, "w") as f:
        json.dump(hist_data, f, indent=2)
    print(f"History : {history_path}")
    print(f"Modèle  : {model_path}")

    # ----------------------------------------------------------------------
    # 8. Plot courbes (style workshop : accuracy à gauche, loss à droite)
    # ----------------------------------------------------------------------
    n_ep = len(history.history["loss"])
    epochs_range = range(1, n_ep + 1)

    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history.history["accuracy"], label="Training Accuracy")
    plt.plot(epochs_range, history.history["val_accuracy"], label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history.history["loss"], label="Training Loss")
    plt.plot(epochs_range, history.history["val_loss"], label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(True, alpha=0.3)

    fig_path = FIGURES_DIR / "022_workshop_curves.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=120)
    plt.close()
    print(f"Courbes : {fig_path}")

    # ----------------------------------------------------------------------
    # 9. Évaluation finale (sur la val, pas de test set séparé ici — style workshop)
    # ----------------------------------------------------------------------
    print("\n4. Évaluation finale sur le set de validation...")
    val_loss, val_acc = model.evaluate(val_set, verbose=1)
    print(f"Val loss     : {val_loss:.4f}")
    print(f"Val accuracy : {val_acc:.4f}")

    print("\n" + "=" * 70)
    print("Terminé.")
    print(f"Modèle  : {model_path}")
    print(f"History : {history_path}")
    print(f"Courbes : {fig_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
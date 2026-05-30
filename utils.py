"""
Fonctions utilitaires partagées entre les notebooks du projet TouNum.

Version optimisée :
- Chargement **complet en mémoire** une seule fois (resize fait une seule fois)
- tf.data.Dataset avec cache + prefetch AUTOTUNE
- Augmentation via keras preprocessing layers (rapide, intégrable au modèle)

Auteur : projet CESI TouNum
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix


# ----------------------------------------------------------------------------
# Config centrale (config.json)
# ----------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).parent / "config.json"
with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    CFG = json.load(_f)

# Constantes exposées
CLASS_NAMES: list[str] = CFG["class_names"]
IMG_SIZE: tuple[int, int] = tuple(CFG["img_size"])
BATCH_SIZE: int = CFG["batch_size"]
SEED: int = CFG["seed"]

DATASET_DIR = Path("Dataset/Dataset")
SPLIT_DIR = Path("data_split")
MODELS_DIR = Path("models")
HISTORIES_DIR = Path("histories")

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


# ----------------------------------------------------------------------------
# Chargement in-memory (clé pour la perf)
# ----------------------------------------------------------------------------

def _list_images(dir_: Path) -> list[Path]:
    return sorted(p for p in dir_.rglob("*")
                  if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS)


def _load_images_as_array(
    dir_: Path, img_size: tuple[int, int], dtype=np.float32
) -> np.ndarray:
    """Charge toutes les images d'un dossier, redimensionnées, en numpy array."""
    paths = _list_images(dir_)
    if not paths:
        return np.empty((0, img_size[1], img_size[0], 3), dtype=dtype)
    images = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB").resize(img_size, Image.BILINEAR)
            arr = np.asarray(img, dtype=dtype)
            if dtype == np.float32:
                arr = arr / 255.0
            images.append(arr)
        except Exception as e:
            print(f"  illisible : {p.name} — {e}")
    return np.stack(images)


def load_split_in_memory(
    split_dir: Path = SPLIT_DIR,
    classes: list[str] | None = None,
    img_size: tuple[int, int] = IMG_SIZE,
    class_mode: str = "categorical",
    use_uint8: bool = False,
) -> dict:
    """Charge train/val/test en mémoire (un seul resize, plus jamais d'I/O après).

    Retourne un dict avec :
        X_train, y_train, X_val, y_val, X_test, y_test
        class_indices, num_classes

    use_uint8=True permet de stocker en uint8 (4× moins de RAM) ; la normalisation
    en [0, 1] est alors faite dans le tf.data pipeline (en float16/32).
    """
    if classes is None:
        classes = sorted(d.name for d in (split_dir / "train").iterdir() if d.is_dir())
    class_indices = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)
    dtype = np.uint8 if use_uint8 else np.float32

    out = {}
    for subset in ["train", "val", "test"]:
        X_parts, y_parts = [], []
        print(f"Chargement {subset}...")
        t0 = time.time()
        total = 0
        for c in classes:
            cdir = split_dir / subset / c
            if not cdir.exists():
                print(f"  {cdir} introuvable, skip")
                continue
            arr = _load_images_as_array(cdir, img_size, dtype=dtype)
            X_parts.append(arr)
            y_parts.append(np.full(len(arr), class_indices[c], dtype=np.int32))
            total += len(arr)
        X = np.concatenate(X_parts) if X_parts else np.empty((0,) + img_size + (3,), dtype=dtype)
        y_int = np.concatenate(y_parts) if y_parts else np.empty((0,), dtype=np.int32)
        if class_mode == "categorical":
            y = tf.keras.utils.to_categorical(y_int, num_classes=num_classes).astype(np.float32)
        else:  # binary
            y = y_int.astype(np.float32)
        out[f"X_{subset}"] = X
        out[f"y_{subset}"] = y
        out[f"y_{subset}_int"] = y_int
        elapsed = time.time() - t0
        size_mb = X.nbytes / 1e6
        print(f"  {subset:5s} : {X.shape} ({size_mb:.1f} MB) en {elapsed:.1f}s")

    out["class_indices"] = class_indices
    out["num_classes"] = num_classes
    out["classes"] = classes
    return out


# ----------------------------------------------------------------------------
# tf.data pipelines
# ----------------------------------------------------------------------------

def make_tf_dataset(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = False,
    cache: bool = True,
    normalize: bool = False,
    seed: int = SEED,
) -> tf.data.Dataset:
    """Crée un tf.data.Dataset optimisé à partir d'arrays numpy.

    normalize=True : divise par 255 dans le pipeline (utile si X est en uint8)
    cache=True : garde le dataset en RAM (gros gain sur epochs 2+)
    """
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if normalize:
        ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(X), 4096), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def make_augmentation_layer() -> tf.keras.Sequential:
    aug = CFG["augmentation"]
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip(aug["flip"]),
        tf.keras.layers.RandomRotation(aug["rotation"]),
        tf.keras.layers.RandomZoom(aug["zoom"]),
        tf.keras.layers.RandomTranslation(aug["translation"], aug["translation"]),
    ], name="augmentation")


# ----------------------------------------------------------------------------
# Sauvegarde / chargement des histories
# ----------------------------------------------------------------------------

def save_history(history: tf.keras.callbacks.History, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {k: [float(x) for x in v] for k, v in history.history.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_history(path: Path | str) -> dict[str, list[float]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------

def plot_history(history_dict: dict[str, list[float]], title: str = "",
                 save_path: Path | str | None = None) -> None:
    epochs = range(1, len(history_dict["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(epochs, history_dict["loss"], label="train loss")
    if "val_loss" in history_dict:
        axes[0].plot(epochs, history_dict["val_loss"], label="val loss")
    axes[0].set_title(f"Loss — {title}"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    acc_key = "accuracy" if "accuracy" in history_dict else "acc"
    val_acc_key = "val_accuracy" if "val_accuracy" in history_dict else "val_acc"
    if acc_key in history_dict:
        axes[1].plot(epochs, history_dict[acc_key], label="train acc")
        if val_acc_key in history_dict:
            axes[1].plot(epochs, history_dict[val_acc_key], label="val acc")
        axes[1].set_title(f"Accuracy — {title}"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
        axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120); plt.close()
    else:
        plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, title="Matrice de confusion",
                          normalize=False, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
    else:
        fmt = "d"
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, cbar=True)
    plt.xlabel("Prédit"); plt.ylabel("Réel"); plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120); plt.close()
    else:
        plt.show()
    return cm


def print_classification_report(y_true, y_pred, class_names):
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))


# ----------------------------------------------------------------------------
# Reproductibilité
# ----------------------------------------------------------------------------

def set_seeds(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)    
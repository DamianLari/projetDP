"""
04 : final benchmark: alone multi-class vs cascade pipeline (multi-class + binaire).

The cascade pipeline : the multi-class model predicts the class ; if the prediction is
Painting or Photo (the most confused classes), the image is sent to the specialized
binary model to make a more precise decision.

All is driven by config_pipeline.json : you just need to change the paths of the
models to benchmark other versions (no code modification necessary).

Usage:
    python3 04_evaluation_benchmark.py

Return:
    - figures/04_*.png
    - logs (classification reports, tableau comparatif)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "03_binary_model"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)

from binary_model import _preprocess_for, BINARY_CLASSES
from utils import SPLIT_DIR, load_history, set_seeds, SEED, CLASS_NAMES, IMG_SIZE, BATCH_SIZE

CFG_PATH = Path(__file__).parent / "config_pipeline.json"
FIGURES_DIR = Path("figures")
AUTOTUNE = tf.data.AUTOTUNE


# Pipelines (same decode/preprocessing for multi-class and binary, so the images are in the same order for both)

def _list_test_files() -> tf.data.Dataset:
    pattern = str(SPLIT_DIR / "test" / "*" / "*")
    return tf.data.Dataset.list_files(pattern, shuffle=False, seed=SEED)


def _decode(path: tf.Tensor, img_h: int, img_w: int) -> tf.Tensor:
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    return tf.image.resize(img, [img_h, img_w])


def _label(path: tf.Tensor) -> tf.Tensor:
    parts = tf.strings.split(path, "/")
    n = tf.shape(parts)[0]
    class_name = parts[n - 2]
    matches = tf.cast(tf.equal(class_name, tf.constant(CLASS_NAMES)), tf.int32)
    return tf.cast(tf.argmax(matches), tf.int32)


def make_multiclass_ds() -> tf.data.Dataset:
    img_h, img_w = IMG_SIZE
    files = _list_test_files()
    ds = files.map(lambda p: (_decode(p, img_h, img_w) / 255.0, _label(p)),
                   num_parallel_calls=AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)


def make_binary_probs_ds(backbone: str) -> tf.data.Dataset:
    img_h, img_w = IMG_SIZE
    preprocess = _preprocess_for({"backbone": backbone})
    files = _list_test_files()
    ds = files.map(lambda p: preprocess(_decode(p, img_h, img_w)),
                   num_parallel_calls=AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)


# Plots

def plot_curves(history_dict: dict, save_path: Path, title: str) -> None:
    epochs = range(1, len(history_dict["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(epochs, history_dict["loss"], label="train loss")
    if "val_loss" in history_dict:
        axes[0].plot(epochs, history_dict["val_loss"], label="val loss")
    axes[0].set_title(f"Loss : {title}"); axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    if "accuracy" in history_dict:
        axes[1].plot(epochs, history_dict["accuracy"], label="train acc")
    if "val_accuracy" in history_dict:
        axes[1].plot(epochs, history_dict["val_accuracy"], label="val acc")
    axes[1].set_title(f"Accuracy : {title}"); axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=120); plt.close()


def plot_cm(y_true, y_pred, class_names, save_path, title, normalize=False):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
    else:
        fmt = "d"
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Prédit"); plt.ylabel("Réel"); plt.title(title)
    plt.tight_layout(); plt.savefig(save_path, dpi=120); plt.close()


# Main

def main() -> None:
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    set_seeds(SEED)
    FIGURES_DIR.mkdir(exist_ok=True)
    t0 = time.time()

    print("=" * 70)
    print("BENCHMARK : multi-class seul vs cascade pipeline (multi-class + binaire)")
    print(f"  Multi-class : {cfg['multiclass_model']}")
    print(f"  Binaire     : {cfg['binary_model']}")
    print("=" * 70)

    # Training curves (optional)
    print("\n1. Training curves...")
    for path, title, out in [
        (Path("histories/multiclass_023_history.json"), "Multi-class", "04_curves_multiclass.png"),
        (Path("histories/binary_photo_painting_history.json"), "Binaire Photo vs Painting", "04_curves_binary.png"),
    ]:
        if path.exists():
            plot_curves(load_history(path), FIGURES_DIR / out, title)
            print(f"  {out}")
        else:
            print(f"  {path} introuvable, skip")

    # Models
    print("\n2. Loading models...")
    multi_model = tf.keras.models.load_model(cfg["multiclass_model"])
    bin_model   = tf.keras.models.load_model(cfg["binary_model"])
    binary_backbone = bin_model.name.replace("_binary", "")
    print(f"  Multi-class params : {multi_model.count_params():,}")
    print(f"  Binary params      : {bin_model.count_params():,}  (backbone : {binary_backbone})")

    # Multi-class predictions
    print("\n3. Multi-class predictions on the test set...")
    test_ds_multi = make_multiclass_ds()
    y_true_all, probs_multi_all = [], []
    for imgs, lbls in test_ds_multi:
        y_true_all.append(lbls.numpy())
        probs_multi_all.append(multi_model(imgs, training=False).numpy())
    y_true = np.concatenate(y_true_all)
    probs_multi = np.concatenate(probs_multi_all)
    y_pred_multi = np.argmax(probs_multi, axis=1)
    print(f"  Accuracy multi-class : {accuracy_score(y_true, y_pred_multi):.4f}")
    print(f"  Test set : {len(y_true)} images")

    plot_cm(y_true, y_pred_multi, CLASS_NAMES, FIGURES_DIR / "04_cm_multiclass_raw.png",
            "Multi-class — brut")
    plot_cm(y_true, y_pred_multi, CLASS_NAMES, FIGURES_DIR / "04_cm_multiclass_norm.png",
            "Multi-class — normalisé", normalize=True)
    print("\nClassification report — multi-class :")
    print(classification_report(y_true, y_pred_multi, target_names=CLASS_NAMES, digits=4))

    # Cascade pipeline
    print("\n4. Cascade pipeline (multi-class → binaire sur Painting/Photo)...")
    test_ds_bin = make_binary_probs_ds(binary_backbone)
    probs_bin = np.concatenate([
        bin_model(imgs, training=False).numpy().flatten() for imgs in test_ds_bin
    ])

    idx_painting = CLASS_NAMES.index(BINARY_CLASSES[0])
    idx_photo    = CLASS_NAMES.index(BINARY_CLASSES[1])

    y_pred_cascade = y_pred_multi.copy()
    mask = np.isin(y_pred_multi, [idx_painting, idx_photo])
    refined = np.where(probs_bin >= 0.5, idx_photo, idx_painting)
    y_pred_cascade[mask] = refined[mask]

    print(f"  Accuracy pipeline cascade : {accuracy_score(y_true, y_pred_cascade):.4f}")
    print(f"  Enhanced images : {mask.sum()} / {len(y_true)}")

    plot_cm(y_true, y_pred_cascade, CLASS_NAMES, FIGURES_DIR / "04_cm_cascade_raw.png",
            "Cascadé — brut")
    plot_cm(y_true, y_pred_cascade, CLASS_NAMES, FIGURES_DIR / "04_cm_cascade_norm.png",
            "Cascadé — normalisé", normalize=True)
    print("\nClassification report — pipeline cascadé :")
    print(classification_report(y_true, y_pred_cascade, target_names=CLASS_NAMES, digits=4))

    # Global comparison
    print("\n5. Global comparison:")
    rows = []
    for name, y_pred in [("Multi-class alone", y_pred_multi), ("Cascade pipeline", y_pred_cascade)]:
        rows.append({
            "modèle": name,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        })
    print(pd.DataFrame(rows).to_string(index=False, float_format="%.4f"))

    # Focus Photo vs Painting
    print("\n6. Focus Photo vs Painting only:")
    mask_pp = np.isin(y_true, [idx_painting, idx_photo])
    yt = y_true[mask_pp]
    print(f"  Multi-class alone : {accuracy_score(yt, y_pred_multi[mask_pp]):.4f}")
    print(f"  Cascade pipeline  : {accuracy_score(yt, y_pred_cascade[mask_pp]):.4f}")

    print("\n" + "=" * 70)
    print(f"Benchmark finished in {time.time()-t0:.1f} s : plots in {FIGURES_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()

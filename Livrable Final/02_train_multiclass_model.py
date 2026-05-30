"""
02 — CNN multi-class 5 classes (modèle 023).

Entraîne le modèle défini dans multiclass_model.py à partir des hyperparamètres
de config_multiclass.json. Pipeline et architecture : voir multiclass_model.py
(aucune duplication ici).

Sortie :
    - models/multiclass_023_best.keras
    - histories/multiclass_023_history.json
    - figures/02_multiclass_curves.png

Usage :
    python3 02_model_multiclass.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Le module multiclass_model et son config vivent dans 02_multiclass_model/
MODEL_DIR = Path(__file__).resolve().parent / "02_multiclass_model"
sys.path.insert(0, str(MODEL_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from multiclass_model import make_dataset, run_training
from utils import CLASS_NAMES, HISTORIES_DIR, MODELS_DIR, SPLIT_DIR, set_seeds

CFG_PATH = MODEL_DIR / "config_multiclass.json"
FIGURES_DIR = Path("figures")


def plot_curves(history_dict: dict, save_path: Path) -> None:
    epochs = range(1, len(history_dict["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(epochs, history_dict["loss"], label="train loss")
    axes[0].plot(epochs, history_dict["val_loss"], label="val loss")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, history_dict["accuracy"], label="train acc")
    axes[1].plot(epochs, history_dict["val_accuracy"], label="val acc")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=120); plt.close()


def main() -> None:
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    set_seeds(cfg["seed"])
    MODELS_DIR.mkdir(exist_ok=True)
    HISTORIES_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print(f"TF version : {tf.__version__}")
    print(f"GPUs       : {tf.config.list_physical_devices('GPU')}")
    print(f"Classes    : {CLASS_NAMES}")
    print(f"IMG_SIZE   : {cfg['img_size']}  BATCH : {cfg['batch_size']}  EPOCHS : {cfg['epochs']}")
    print("=" * 70)

    if not SPLIT_DIR.exists():
        raise SystemExit(f"ERREUR : {SPLIT_DIR} introuvable. Lance d'abord 01_data_split_analysis.py.")

    print("\n1. Construction des pipelines tf.data...")
    train_ds = make_dataset("train", cfg, augment=True)
    val_ds   = make_dataset("val",   cfg, augment=False)
    test_ds  = make_dataset("test",  cfg, augment=False)
    print("Pipelines prêts.")

    model_path   = MODELS_DIR / "multiclass_023_best.keras"
    history_path = HISTORIES_DIR / "multiclass_023_history.json"

    print("\n2. Entraînement...")
    model, history = run_training(cfg, train_ds, val_ds,
                                  model_path=model_path, verbose=1)

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump({k: [float(x) for x in v] for k, v in history.items()}, f, indent=2)
    print(f"\nHistory : {history_path}")
    print(f"Modèle  : {model_path}")

    fig_path = FIGURES_DIR / "02_multiclass_curves.png"
    plot_curves(history, fig_path)
    print(f"Courbes : {fig_path}")

    print("\n3. Évaluation sur le test set...")
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"Test loss     : {test_loss:.4f}")
    print(f"Test accuracy : {test_acc:.4f}")

    print("\n" + "=" * 70)
    print(f"Terminé. Modèle : {model_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

"""
03 — CNN binaire Photo vs Painting (transfer learning, 2 phases).

Entraîne le modèle défini dans binary_model.py à partir des hyperparamètres
de config_binary.json. Utilisé en aval du multi-class 023 pour raffiner la
distinction Photo / Painting (classes les plus confondues).

Architecture et pipeline : voir binary_model.py (aucune duplication ici).

Sortie :
    - models/binary_photo_painting_best.keras
    - histories/binary_photo_painting_history.json
    - figures/03_binary_curves.png

Usage :
    python3 03_model_binary_photo_vs_painting.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from binary_model import BINARY_CLASSES, make_dataset, run_training
from utils import HISTORIES_DIR, MODELS_DIR, set_seeds

CFG_PATH = Path(__file__).parent / "config_binary.json"
FIGURES_DIR = Path("figures")


def plot_curves(history_dict: dict, save_path: Path, title: str = "Binaire") -> None:
    epochs = range(1, len(history_dict["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(epochs, history_dict["loss"], label="train loss")
    axes[0].plot(epochs, history_dict["val_loss"], label="val loss")
    axes[0].set_title(f"Loss — {title}"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, history_dict["accuracy"], label="train acc")
    axes[1].plot(epochs, history_dict["val_accuracy"], label="val acc")
    axes[1].set_title(f"Accuracy — {title}"); axes[1].set_xlabel("Epoch")
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
    print(f"Backbone   : {cfg['backbone']}   Classes : {BINARY_CLASSES}")
    print(f"IMG_SIZE   : {cfg['img_size']}  BATCH : {cfg['batch_size']}")
    print("=" * 70)

    print("\n1. Construction des pipelines tf.data...")
    train_ds = make_dataset("train", cfg, augment=True)
    val_ds   = make_dataset("val",   cfg, augment=False)
    test_ds  = make_dataset("test",  cfg, augment=False)
    print("Pipelines prêts.")

    model_path   = MODELS_DIR / "binary_photo_painting_best.keras"
    history_path = HISTORIES_DIR / "binary_photo_painting_history.json"

    print("\n2. Entraînement deux phases (tête + fine-tuning)...")
    model, history = run_training(cfg, train_ds, val_ds,
                                  model_path=model_path, verbose=1)

    # save_history attend un objet History ; on a un dict → sauvegarde directe
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump({k: [float(x) for x in v] for k, v in history.items()}, f, indent=2)
    print(f"\nHistory : {history_path}")
    print(f"Modèle  : {model_path}")

    fig_path = FIGURES_DIR / "03_binary_curves.png"
    plot_curves(history, fig_path, title=f"Binaire {cfg['backbone']} Photo vs Painting")
    print(f"Courbes : {fig_path}")

    print("\n3. Évaluation sur le test set...")
    results = model.evaluate(test_ds, verbose=1)
    for name, val in zip(model.metrics_names, results):
        print(f"  {name:10s} : {val:.4f}")

    print("\n" + "=" * 70)
    print(f"Terminé. Modèle : {model_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

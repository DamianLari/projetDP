"""
analyze_tune_results.py — Analyse complète des modèles tunés (tune_results/)

Pour chaque modèle :
  - model.summary()
  - Benchmark sur le jeu de test (accuracy, loss, recall par classe)

Graphiques produits :
  1. Bar chart : accuracy / loss / recall macro par modèle
  2. Scatter : accuracy vs nb de paramètres (rapport perf/légèreté)
  3. Bar chart importance des hyperparamètres (corrélation avec test_acc)

Usage :
    python3 analyze_tune_results.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)

# Schéma commun aux benchmark_results.json (multiclass ET binaire)
BENCHMARK_SCHEMA = [
    "label", "trial_id", "backbone", "test_acc", "test_loss", "test_auc",
    "precision_macro", "recall_macro", "f1_macro",
    "precision_per_class", "recall_per_class", "f1_per_class",
    "n_params", "n_params_deploy", "n_params_train", "meets_target", "config",
]

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).parent
RESULTS_DIR = ROOT / "tune_multiclass_results"
SPLIT_DIR   = ROOT / "data_split"
FIGURES_DIR = RESULTS_DIR / "analysis"
FIGURES_DIR.mkdir(exist_ok=True)

CLASS_NAMES = ["Painting", "Photo", "Schematics", "Sketch", "Text"]
IMG_SIZE    = (224, 224)
AUTOTUNE    = tf.data.AUTOTUNE


# ---------------------------------------------------------------------------
# Dataset de test
# ---------------------------------------------------------------------------

def _decode_img(path: tf.Tensor) -> tf.Tensor:
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, list(IMG_SIZE))
    return tf.cast(img, tf.float32) / 255.0


def _get_label(path: tf.Tensor) -> tf.Tensor:
    parts = tf.strings.split(path, "/")
    n = tf.shape(parts)[0]
    class_name = parts[n - 2]
    matches = tf.cast(tf.equal(class_name, tf.constant(CLASS_NAMES)), tf.int32)
    return tf.cast(tf.argmax(matches), tf.int32)


def make_test_dataset(batch_size: int = 32) -> tf.data.Dataset:
    pattern = str(SPLIT_DIR / "test" / "*" / "*")
    ds = tf.data.Dataset.list_files(pattern, shuffle=False)
    ds = ds.map(lambda p: (_decode_img(p), _get_label(p)), num_parallel_calls=AUTOTUNE)
    return ds.batch(batch_size).prefetch(AUTOTUNE)


# ---------------------------------------------------------------------------
# Benchmark d'un modèle
# ---------------------------------------------------------------------------

def benchmark_model(model: tf.keras.Model, test_ds: tf.data.Dataset) -> dict:
    loss, acc = model.evaluate(test_ds, verbose=0)

    y_true, y_pred, y_prob = [], [], []
    for x_batch, y_batch in test_ds:
        preds = model(x_batch, training=False).numpy()
        y_true.extend(y_batch.numpy().tolist())
        y_pred.extend(np.argmax(preds, axis=1).tolist())
        y_prob.extend(preds.tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    # AUC macro one-vs-rest (None si une classe est absente du jeu de test)
    try:
        test_auc = round(float(roc_auc_score(
            y_true, y_prob, multi_class="ovr", average="macro")), 4)
    except ValueError:
        test_auc = None

    recall_macro    = float(recall_score(y_true, y_pred, average="macro",    zero_division=0))
    precision_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro        = float(f1_score(y_true, y_pred, average="macro",        zero_division=0))
    recall_per_class    = recall_score(y_true, y_pred, average=None, zero_division=0).tolist()
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0).tolist()
    f1_per_class        = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()

    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, digits=4, output_dict=False
    )
    return {
        "test_acc":        round(acc, 4),
        "test_loss":       round(loss, 4),
        "test_auc":        test_auc,
        "recall_macro":    round(recall_macro, 4),
        "precision_macro": round(precision_macro, 4),
        "f1_macro":        round(f1_macro, 4),
        "recall_per_class":    {c: round(r, 4) for c, r in zip(CLASS_NAMES, recall_per_class)},
        "precision_per_class": {c: round(p, 4) for c, p in zip(CLASS_NAMES, precision_per_class)},
        "f1_per_class":        {c: round(f, 4) for c, f in zip(CLASS_NAMES, f1_per_class)},
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "classification_report": report,
    }


# ---------------------------------------------------------------------------
# Importance des hyperparamètres (corrélation de Pearson avec test_acc)
# ---------------------------------------------------------------------------

def compute_hp_importance(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        cfg = r["config"]
        rows.append({
            "test_acc":              r["test_acc"],
            "learning_rate":         cfg["learning_rate"],
            "batch_size":            cfg["batch_size"],
            "dense_units":           cfg["dense_units"],
            "n_blocs":               len(cfg["filters"]),
            "base_filters":          cfg["filters"][0],
            "dropout_bloc":          cfg["dropout"]["bloc1"],
            "dropout_head":          cfg["dropout"]["head"],
            "l2":                    cfg["l2"],
            "reduce_lr_patience":    cfg["reduce_lr_patience"],
            "early_stopping_patience": cfg["early_stopping_patience"],
            "aug_rotation":          cfg["augmentation"]["rotation"],
            "aug_zoom":              cfg["augmentation"]["zoom"],
            "shuffle_buffer":        cfg["shuffle_buffer"],
        })
    df = pd.DataFrame(rows)
    hp_cols = [c for c in df.columns if c != "test_acc"]
    corr = df[hp_cols].corrwith(df["test_acc"]).abs().sort_values(ascending=False)
    return corr


# ---------------------------------------------------------------------------
# Graphiques
# ---------------------------------------------------------------------------

def plot_bar_metrics(df: pd.DataFrame, save_path: Path) -> None:
    """Bar chart : accuracy / loss / recall macro pour chaque modèle."""
    n  = len(df)
    x  = np.arange(n)
    w  = 0.28
    colors_ok   = ["#2ecc71" if m else "#e74c3c" for m in df["meets_target"]]

    fig, axes = plt.subplots(1, 3, figsize=(max(14, n * 0.9), 5))
    fig.suptitle("Benchmark des modèles tunés — jeu de test", fontsize=14, fontweight="bold")

    # Accuracy
    ax = axes[0]
    bars = ax.bar(x, df["test_acc"], color=colors_ok, alpha=0.85, edgecolor="white")
    ax.axhline(0.90, color="green", linestyle="--", linewidth=1.2, label="Cible 90 %")
    for bar, v in zip(bars, df["test_acc"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_title("Test Accuracy"); ax.set_ylabel("Accuracy")
    ax.set_ylim(max(0, df["test_acc"].min() - 0.05), 1.0)
    ax.set_xticks(x); ax.set_xticklabels(df["label"], rotation=45, ha="right")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    # Loss
    ax = axes[1]
    bars = ax.bar(x, df["test_loss"], color=colors_ok, alpha=0.85, edgecolor="white")
    ax.axhline(0.25, color="orange", linestyle="--", linewidth=1.2, label="Cible 0.25")
    for bar, v in zip(bars, df["test_loss"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_title("Test Loss"); ax.set_ylabel("Loss")
    ax.set_xticks(x); ax.set_xticklabels(df["label"], rotation=45, ha="right")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    # Recall macro
    ax = axes[2]
    bars = ax.bar(x, df["recall_macro"], color=colors_ok, alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, df["recall_macro"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_title("Recall macro"); ax.set_ylabel("Recall")
    ax.set_ylim(max(0, df["recall_macro"].min() - 0.05), 1.0)
    ax.set_xticks(x); ax.set_xticklabels(df["label"], rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)

    # Légende commune
    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor="#2ecc71", label="Cible atteinte"),
                    Patch(facecolor="#e74c3c", label="Cible non atteinte")]
    fig.legend(handles=legend_elems, loc="lower right", fontsize=9, ncol=2)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(save_path, dpi=130)
    plt.close()
    print(f"  Sauvegardé : {save_path}")


def _scatter_acc_vs_params(ax, df: pd.DataFrame, param_col: str, xlabel: str) -> None:
    """Sous-graphe scatter accuracy vs une colonne de paramètres."""
    colors_ok = ["#2ecc71" if m else "#e74c3c" for m in df["meets_target"]]
    sc = ax.scatter(
        df[param_col], df["test_acc"],
        c=df["test_loss"], cmap="RdYlGn_r",
        s=120, vmin=0.15, vmax=0.40, zorder=3, edgecolors="grey", linewidths=0.5
    )
    plt.colorbar(sc, ax=ax, label="Test Loss")

    for _, row in df.iterrows():
        ax.annotate(row["label"], (row[param_col], row["test_acc"]),
                    textcoords="offset points", xytext=(6, 3), fontsize=8)

    df_valid = df[df["meets_target"]]
    if not df_valid.empty:
        score = df_valid["test_acc"] / np.log1p(df_valid[param_col])
        best_row = df_valid.loc[score.idxmax()]
        ax.scatter(best_row[param_col], best_row["test_acc"],
                   s=320, facecolors="none", edgecolors="gold", linewidths=2.5,
                   label=f"Meilleur rapport ({best_row['label']})", zorder=4)
        ax.legend(fontsize=9)

    ax.axhline(0.90, color="green", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Test Accuracy", fontsize=10)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1e6:.3f} M"))
    ax.grid(True, alpha=0.3)


def plot_accuracy_vs_params(df: pd.DataFrame, save_path: Path) -> None:
    """Deux scatters côte à côte : params déployés vs params entraînement."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Accuracy vs nombre de paramètres\n"
        "Gauche : modèle déployé (trainable+non-trainable) — "
        "Droite : entraînement (+ état optimiseur Adam)\n"
        "En haut à gauche = léger ET performant  |  ⭕ = meilleur rapport",
        fontsize=11, fontweight="bold"
    )

    _scatter_acc_vs_params(
        axes[0], df,
        param_col="n_params_deploy",
        xlabel="Params déployés (trainable + non-trainable)"
    )
    axes[0].set_title("Vue déploiement", fontsize=11)

    _scatter_acc_vs_params(
        axes[1], df,
        param_col="n_params_train",
        xlabel="Params entraînement (+ état optimiseur Adam)"
    )
    axes[1].set_title("Vue entraînement (RAM GPU)", fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()
    print(f"  Sauvegardé : {save_path}")


def plot_hp_importance(corr: pd.Series, save_path: Path) -> None:
    """Bar chart de l'importance des hyperparamètres (corrélation |Pearson| avec test_acc)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Influence des hyperparamètres sur test_accuracy\n(corrélation |Pearson|)",
                 fontsize=13, fontweight="bold")

    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(corr)))[::-1]
    bars = ax.barh(corr.index[::-1], corr.values[::-1], color=colors[::-1], edgecolor="white")
    for bar, v in zip(bars, corr.values[::-1]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=8)

    ax.set_xlabel("|Corrélation de Pearson| avec test_accuracy", fontsize=10)
    ax.set_xlim(0, min(1, corr.max() + 0.1))
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()
    print(f"  Sauvegardé : {save_path}")


def plot_cm(y_true, y_pred, trial_label: str, save_dir: Path, normalize: bool = False) -> None:
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt, suffix = ".2f", "norm"
    else:
        fmt, suffix = "d", "raw"
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Prédit"); plt.ylabel("Réel")
    plt.title(f"Matrice de confusion — {trial_label} ({'normalisée' if normalize else 'brute'})")
    plt.tight_layout()
    out = save_dir / f"cm_{trial_label}_{suffix}.png"
    plt.savefig(out, dpi=110); plt.close()


def plot_training_curves(history: dict, trial_label: str, save_path: Path,
                         meets_target: bool = True) -> None:
    epochs = range(1, len(history["loss"]) + 1)
    color = "green" if meets_target else "red"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Courbes d'entraînement — {trial_label}", fontsize=12,
                 fontweight="bold", color=color)

    axes[0].plot(epochs, history["loss"], label="train loss")
    axes[0].plot(epochs, history["val_loss"], label="val loss")
    axes[0].axhline(0.25, color="orange", linestyle="--", alpha=0.6, label="Cible 0.25")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["accuracy"], label="train acc")
    axes[1].plot(epochs, history["val_accuracy"], label="val acc")
    axes[1].axhline(0.90, color="green", linestyle="--", alpha=0.6, label="Cible 0.90")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=110); plt.close()


def plot_recall_per_class(df: pd.DataFrame, save_path: Path) -> None:
    """Heatmap recall par classe et par modèle."""
    recall_data = np.array([
        [row[f"recall_{c}"] for c in CLASS_NAMES]
        for _, row in df.iterrows()
    ])

    fig, ax = plt.subplots(figsize=(max(10, len(df) * 0.7), 4))
    fig.suptitle("Recall par classe et par modèle", fontsize=13, fontweight="bold")

    im = ax.imshow(recall_data.T, aspect="auto", cmap="RdYlGn", vmin=0.7, vmax=1.0)
    plt.colorbar(im, ax=ax, label="Recall")

    ax.set_xticks(range(len(df))); ax.set_xticklabels(df["label"], rotation=45, ha="right")
    ax.set_yticks(range(len(CLASS_NAMES))); ax.set_yticklabels(CLASS_NAMES)

    for i in range(len(df)):
        for j in range(len(CLASS_NAMES)):
            v = recall_data[i, j]
            color = "white" if v < 0.82 else "black"
            ax.text(i, j, f"{v:.2f}", ha="center", va="center", fontsize=7, color=color)

    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()
    print(f"  Sauvegardé : {save_path}")


# ---------------------------------------------------------------------------
# Helper console
# ---------------------------------------------------------------------------

def _print_best(best_row: pd.Series, bench_results: list[dict], param_col: str) -> None:
    tid = int(best_row["trial_id"])
    cfg = next(b["config"] for b in bench_results if b["trial_id"] == tid)
    print(f"  Modèle           : {best_row['label']}")
    print(f"  Test acc         : {best_row['test_acc']:.4f}")
    print(f"  Test loss        : {best_row['test_loss']:.4f}")
    print(f"  Recall macro     : {best_row['recall_macro']:.4f}")
    print(f"  F1 macro         : {best_row['f1_macro']:.4f}")
    print(f"  Params déployés  : {int(best_row['n_params_deploy']):,}")
    print(f"  Params entraînem.: {int(best_row['n_params_train']):,}")
    print(f"  Filtres          : {cfg['filters']}")
    print(f"  Dense units      : {cfg['dense_units']}")
    print(f"  Learning rate    : {cfg['learning_rate']}")
    print(f"  Dropout bloc/head: {cfg['dropout']['bloc1']} / {cfg['dropout']['head']}")
    print(f"  L2               : {cfg['l2']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    results_path = RESULTS_DIR / "results.json"
    if not results_path.exists():
        sys.exit(f"Fichier introuvable : {results_path}")

    with open(results_path) as f:
        results: list[dict] = json.load(f)

    print(f"\n{'='*70}")
    print(f"  {len(results)} modèles trouvés dans tune_results/")
    print(f"{'='*70}\n")

    # Jeu de test partagé
    print("Chargement du jeu de test…")
    test_ds = make_test_dataset(batch_size=32)

    bench_results: list[dict] = []

    for r in results:
        trial_id = r["trial_id"]
        model_path = RESULTS_DIR / f"model_trial_{trial_id + 1:02d}.keras"

        print(f"\n{'─'*60}")
        print(f"Trial {trial_id + 1:02d}  —  {model_path.name}")
        print(f"{'─'*60}")

        if not model_path.exists():
            print(f"  ⚠ Fichier manquant : {model_path}")
            continue

        model = tf.keras.models.load_model(str(model_path))

        # Paramètres déployés  = trainable + non-trainable (poids du réseau seul)
        n_deployed = model.count_params()
        # Paramètres entraînement = + état optimiseur (moments Adam m, v, step)
        n_training = sum(int(tf.size(v).numpy()) for v in model.variables)

        # --- summary ---
        print("  ▶ model.summary()")
        model.summary(line_length=80)
        print(f"  Params déployés   : {n_deployed:,}  (trainable + non-trainable)")
        print(f"  Params entraînement: {n_training:,}  (+ état optimiseur Adam)")

        # --- benchmark ---
        print("  ▶ Benchmark sur le jeu de test…")
        bench = benchmark_model(model, test_ds)
        bench["label"]           = f"T{trial_id + 1:02d}"
        bench["trial_id"]        = trial_id
        bench["backbone"]        = "cnn_custom"
        bench["n_params"]        = r["n_params"]      # depuis results.json (cohérence)
        bench["n_params_deploy"] = n_deployed
        bench["n_params_train"]  = n_training
        bench["meets_target"]    = r["meets_target"]
        bench["config"]          = r["config"]
        bench["history"]         = r.get("history", {})
        bench_results.append(bench)

        print(f"  test_acc={bench['test_acc']}  test_loss={bench['test_loss']}  "
              f"recall_macro={bench['recall_macro']}  f1_macro={bench['f1_macro']}")
        print("  Recall / Precision / F1 par classe :")
        for cls in CLASS_NAMES:
            print(f"    {cls:<15}  R={bench['recall_per_class'][cls]:.4f}"
                  f"  P={bench['precision_per_class'][cls]:.4f}"
                  f"  F1={bench['f1_per_class'][cls]:.4f}")

        print("\n  Rapport de classification complet :")
        print(bench["classification_report"])

        # --- matrices de confusion ---
        label = f"T{trial_id+1:02d}"
        cm_dir = FIGURES_DIR / "confusion_matrices"
        cm_dir.mkdir(exist_ok=True)
        y_true_arr = np.array(bench["y_true"])
        y_pred_arr = np.array(bench["y_pred"])
        plot_cm(y_true_arr, y_pred_arr, label, cm_dir, normalize=False)
        plot_cm(y_true_arr, y_pred_arr, label, cm_dir, normalize=True)

        # --- courbes d'entraînement ---
        if bench["history"]:
            curves_dir = FIGURES_DIR / "training_curves"
            curves_dir.mkdir(exist_ok=True)
            plot_training_curves(
                bench["history"], label,
                curves_dir / f"curves_{label}.png",
                meets_target=r["meets_target"],
            )

        tf.keras.backend.clear_session()

    # -----------------------------------------------------------------------
    # DataFrame de synthèse
    # -----------------------------------------------------------------------
    rows = []
    for b in bench_results:
        row = {
            "label":              f"T{b['trial_id']+1:02d}",
            "trial_id":           b["trial_id"],
            "test_acc":           b["test_acc"],
            "test_loss":          b["test_loss"],
            "precision_macro":    b["precision_macro"],
            "recall_macro":       b["recall_macro"],
            "f1_macro":           b["f1_macro"],
            "n_params_deploy":    b["n_params_deploy"],
            "n_params_train":     b["n_params_train"],
            "n_params_deploy_M":  b["n_params_deploy"] / 1e6,
            "n_params_train_M":   b["n_params_train"] / 1e6,
            "meets_target":       b["meets_target"],
        }
        for cls in CLASS_NAMES:
            row[f"recall_{cls}"]    = b["recall_per_class"][cls]
            row[f"precision_{cls}"] = b["precision_per_class"][cls]
            row[f"f1_{cls}"]        = b["f1_per_class"][cls]
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("trial_id").reset_index(drop=True)

    print(f"\n{'='*70}")
    print("TABLEAU DE SYNTHÈSE")
    print("=" * 70)
    display_cols = ["label", "test_acc", "test_loss", "precision_macro",
                    "recall_macro", "f1_macro", "n_params_deploy", "n_params_train", "meets_target"]
    print(df[display_cols].to_string(index=False, float_format="%.4f"))

    # -----------------------------------------------------------------------
    # Meilleur rapport accuracy / paramètres (les deux visions)
    # -----------------------------------------------------------------------
    df_valid = df[df["meets_target"]]
    if not df_valid.empty:
        # Score déployé
        score_dep = df_valid["test_acc"] / np.log1p(df_valid["n_params_deploy_M"])
        best_dep  = df_valid.loc[score_dep.idxmax()]
        # Score entraînement
        score_tr  = df_valid["test_acc"] / np.log1p(df_valid["n_params_train_M"])
        best_tr   = df_valid.loc[score_tr.idxmax()]

        print(f"\n{'='*70}")
        print("MEILLEUR RAPPORT ACCURACY / PARAMS DÉPLOYÉS  (trainable + non-trainable)")
        print(f"{'='*70}")
        _print_best(best_dep, bench_results, "n_params_deploy")

        print(f"\n{'='*70}")
        print("MEILLEUR RAPPORT ACCURACY / PARAMS ENTRAÎNEMENT  (+ état optimiseur Adam)")
        print(f"{'='*70}")
        _print_best(best_tr, bench_results, "n_params_train")
    else:
        print("\n⚠ Aucun modèle n'a atteint la cible.")

    # -----------------------------------------------------------------------
    # Graphiques
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("GÉNÉRATION DES GRAPHIQUES")
    print(f"{'='*70}")

    plot_bar_metrics(df, FIGURES_DIR / "1_bar_acc_loss_recall.png")
    print(f"  Matrices de confusion dans : {FIGURES_DIR}/confusion_matrices/")
    print(f"  Courbes d'entraînement dans : {FIGURES_DIR}/training_curves/")
    plot_accuracy_vs_params(df, FIGURES_DIR / "2_scatter_acc_vs_params.png")
    plot_recall_per_class(df, FIGURES_DIR / "3_heatmap_recall_per_class.png")

    # Importance HP
    corr = compute_hp_importance(results)
    print("\nInfluence des hyperparamètres (|Pearson| avec test_acc) :")
    for hp, c in corr.items():
        bar = "█" * int(c * 30)
        print(f"  {hp:<30} {c:.3f}  {bar}")
    plot_hp_importance(corr, FIGURES_DIR / "4_hp_importance.png")

    # -----------------------------------------------------------------------
    # Sauvegarde des métriques enrichies
    # -----------------------------------------------------------------------
    out_path = RESULTS_DIR / "benchmark_results.json"
    # Sérialisation selon le schéma commun (ordre et champs identiques au binaire)
    clean_bench = [{k: b.get(k) for k in BENCHMARK_SCHEMA} for b in bench_results]
    with open(out_path, "w") as f:
        json.dump(clean_bench, f, indent=2)
    print(f"\nMétriques enrichies sauvegardées : {out_path}")
    print(f"Graphiques dans : {FIGURES_DIR}/")

    # Configs prêtes à l'emploi : meilleure accuracy & meilleur rapport perf/taille
    export_best_configs(bench_results)


def export_best_configs(bench_results: list[dict]) -> None:
    """Écrit best_acc_config.json et best_efficiency_config.json (format config_multiclass.json),
    directement réutilisables sans copier-coller manuel."""
    if not bench_results:
        return

    # Meilleure accuracy (tie-break : loss la plus basse)
    best_acc = max(bench_results, key=lambda b: (b["test_acc"], -b["test_loss"]))

    # Meilleur rapport perf/taille parmi ceux qui atteignent la cible (sinon tous).
    # Taille = nb de params TOTAL avec état optimiseur Adam (n_params_train).
    pool = [b for b in bench_results if b["meets_target"]] or bench_results
    best_eff = max(pool, key=lambda b: b["test_acc"] / np.log1p(b["n_params_train"] / 1e6))

    acc_path = RESULTS_DIR / "best_acc_config.json"
    eff_path = RESULTS_DIR / "best_efficiency_config.json"
    with open(acc_path, "w") as f:
        json.dump(best_acc["config"], f, indent=2)
    with open(eff_path, "w") as f:
        json.dump(best_eff["config"], f, indent=2)

    print(f"\n{'='*70}\nCONFIGS EXPORTÉES (prêtes pour config_multiclass.json)\n{'='*70}")
    print(f"  Meilleure accuracy   : {best_acc['label']}  "
          f"acc={best_acc['test_acc']}  loss={best_acc['test_loss']}")
    print(f"    → {acc_path}")
    print(f"  Meilleur rapport     : {best_eff['label']}  "
          f"acc={best_eff['test_acc']}  params_train={int(best_eff['n_params_train']):,}")
    print(f"    → {eff_path}")


if __name__ == "__main__":
    main()

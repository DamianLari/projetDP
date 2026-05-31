"""
analyze_binary_tune_results.py : Analyse complète des modèles binaires tunés.

Pour chaque modèle de tune_binary_results/ :
  - model.summary()
  - Benchmark sur le jeu de test (accuracy, loss, AUC, precision, recall, F1)
  - Matrice de confusion 2×2

Graphiques produits (dans tune_binary_results/analysis/) :
  1. Bar chart : accuracy / AUC / F1 par modèle
  2. Scatter : accuracy vs nb de paramètres entraînables
  3. Heatmap recall par classe
  4. Importance des hyperparamètres (|corrélation| avec test_acc)
  + matrices de confusion par modèle

Usage :
    python3 analyze_binary_tune_results.py
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

from binary_model import BINARY_CLASSES, make_dataset

ROOT        = Path(__file__).parent
RESULTS_DIR = ROOT / "tune_binary_results"
FIGURES_DIR = RESULTS_DIR / "analysis"
FIGURES_DIR.mkdir(exist_ok=True)

# Seuils de qualité (alignés sur tune_binary.py) : acc ≥ 0.96 ET loss ≤ 0.10
TARGET_ACC  = 0.96
TARGET_LOSS = 0.10

# Schéma commun aux benchmark_results.json (multiclass ET binaire)
BENCHMARK_SCHEMA = [
    "label", "trial_id", "backbone", "test_acc", "test_loss", "test_auc",
    "precision_macro", "recall_macro", "f1_macro",
    "precision_per_class", "recall_per_class", "f1_per_class",
    "n_params", "n_params_deploy", "n_params_train", "meets_target", "config",
]


# ---------------------------------------------------------------------------
# Benchmark d'un modèle binaire
# ---------------------------------------------------------------------------

def benchmark_model(model: tf.keras.Model, cfg: dict) -> dict:
    test_ds = make_dataset("test", cfg, augment=False)
    loss = model.evaluate(test_ds, verbose=0, return_dict=True).get("loss", 0.0)

    y_true, y_prob = [], []
    for x_batch, y_batch in test_ds:
        probs = model(x_batch, training=False).numpy().flatten()
        y_true.extend(y_batch.numpy().tolist())
        y_prob.extend(probs.tolist())

    y_true = np.array(y_true).astype(int)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    return {
        "test_acc":   round(float(accuracy_score(y_true, y_pred)), 4),
        "test_loss":  round(float(loss), 4),
        "test_auc":   round(float(roc_auc_score(y_true, y_prob)), 4),
        "precision_macro": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "recall_macro":    round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_macro":        round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "recall_per_class": {c: round(float(r), 4) for c, r in
                             zip(BINARY_CLASSES, recall_score(y_true, y_pred, average=None, zero_division=0))},
        "precision_per_class": {c: round(float(p), 4) for c, p in
                                zip(BINARY_CLASSES, precision_score(y_true, y_pred, average=None, zero_division=0))},
        "f1_per_class": {c: round(float(f), 4) for c, f in
                         zip(BINARY_CLASSES, f1_score(y_true, y_pred, average=None, zero_division=0))},
        "report": classification_report(y_true, y_pred, target_names=BINARY_CLASSES, digits=4),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }


# ---------------------------------------------------------------------------
# Importance des hyperparamètres
# ---------------------------------------------------------------------------

def compute_hp_importance(results: list[dict]) -> pd.Series:
    rows = []
    for r in results:
        cfg = r["config"]
        rows.append({
            "test_acc":          r["test_acc"],
            "backbone":          0 if cfg["backbone"] == "MobileNetV2" else 1,
            "lr_finetune":       cfg["lr_finetune"],
            "lr_head":           cfg["lr_head"],
            "unfreeze_fraction": cfg["unfreeze_fraction"],
            "dense_units":       cfg["dense_units"],
            "dropout_head":      cfg["dropout_head"],
            "batch_size":        cfg["batch_size"],
            "aug_brightness":    cfg["augmentation"]["brightness"],
            "aug_zoom":          cfg["augmentation"]["zoom"],
        })
    df = pd.DataFrame(rows)
    hp_cols = [c for c in df.columns if c != "test_acc"]
    return df[hp_cols].corrwith(df["test_acc"]).abs().sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Graphiques
# ---------------------------------------------------------------------------

def plot_bar_metrics(df: pd.DataFrame, save_path: Path) -> None:
    n, x = len(df), np.arange(len(df))
    fig, axes = plt.subplots(1, 3, figsize=(max(14, n * 0.9), 5))
    fig.suptitle("Benchmark des modèles binaires tunés : jeu de test", fontsize=14, fontweight="bold")
    for ax, col, title, cible in [
        (axes[0], "test_acc", "Test Accuracy", 0.95),
        (axes[1], "test_auc", "Test AUC", None),
        (axes[2], "f1_macro", "F1 macro", None),
    ]:
        bars = ax.bar(x, df[col], color="steelblue", alpha=0.85, edgecolor="white")
        if cible:
            ax.axhline(cible, color="green", linestyle="--", linewidth=1.2, label=f"Cible {cible}")
            ax.legend(fontsize=8)
        for bar, v in zip(bars, df[col]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_title(title); ax.set_ylim(min(0.8, df[col].min() - 0.05), 1.0)
        ax.set_xticks(x); ax.set_xticklabels(df["label"], rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=130); plt.close()
    print(f"  Sauvegardé : {save_path}")


def plot_bar_acc_loss_recall(df: pd.DataFrame, save_path: Path) -> None:
    """Bar chart accuracy / loss / recall macro, coloré selon meets_target.

    Permet de repérer en un coup d'œil les meilleurs modèles (barres vertes =
    cible atteinte : acc ≥ 0.96 ET loss ≤ 0.10)."""
    n  = len(df)
    x  = np.arange(n)
    colors_ok = ["#2ecc71" if m else "#e74c3c" for m in df["meets_target"]]

    fig, axes = plt.subplots(1, 3, figsize=(max(14, n * 0.9), 5))
    fig.suptitle("Benchmark des modèles binaires tunés : jeu de test",
                 fontsize=14, fontweight="bold")

    # Accuracy
    ax = axes[0]
    bars = ax.bar(x, df["test_acc"], color=colors_ok, alpha=0.85, edgecolor="white")
    ax.axhline(TARGET_ACC, color="green", linestyle="--", linewidth=1.2,
               label=f"Cible {TARGET_ACC}")
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
    ax.axhline(TARGET_LOSS, color="orange", linestyle="--", linewidth=1.2,
               label=f"Cible {TARGET_LOSS}")
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

    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor="#2ecc71", label="Cible atteinte"),
                    Patch(facecolor="#e74c3c", label="Cible non atteinte")]
    fig.legend(handles=legend_elems, loc="lower right", fontsize=9, ncol=2)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(save_path, dpi=130); plt.close()
    print(f"  Sauvegardé : {save_path}")


def plot_acc_vs_params(df: pd.DataFrame, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Accuracy vs nb de paramètres entraînables\n(en haut à gauche = léger ET performant)",
                 fontsize=13, fontweight="bold")
    sc = ax.scatter(df["n_params_train"], df["test_acc"], c=df["test_loss"], cmap="RdYlGn_r",
                    s=120, zorder=3, edgecolors="grey", linewidths=0.5)
    plt.colorbar(sc, ax=ax, label="Test Loss")
    for _, row in df.iterrows():
        ax.annotate(f"{row['label']}\n{row['backbone']}", (row["n_params_train"], row["test_acc"]),
                    textcoords="offset points", xytext=(6, 3), fontsize=7)
    score = df["test_acc"] / np.log1p(df["n_params_train"] / 1e6)
    best = df.loc[score.idxmax()]
    ax.scatter(best["n_params_train"], best["test_acc"], s=320, facecolors="none",
               edgecolors="gold", linewidths=2.5, label=f"Meilleur rapport ({best['label']})", zorder=4)
    ax.legend(fontsize=9)
    ax.set_xlabel("Paramètres totaux (+ état optimiseur Adam)"); ax.set_ylabel("Test Accuracy")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1e6:.2f} M"))
    ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=130); plt.close()
    print(f"  Sauvegardé : {save_path}")


def plot_hp_importance(corr: pd.Series, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Influence des hyperparamètres sur test_accuracy\n(corrélation |Pearson|)",
                 fontsize=13, fontweight="bold")
    bars = ax.barh(corr.index[::-1], corr.values[::-1], color="steelblue", edgecolor="white")
    for bar, v in zip(bars, corr.values[::-1]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlabel("|Corrélation de Pearson| avec test_accuracy")
    ax.set_xlim(0, min(1, corr.max() + 0.1)); ax.grid(axis="x", alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=130); plt.close()
    print(f"  Sauvegardé : {save_path}")


def plot_cm(y_true, y_pred, label: str, save_dir: Path, normalize: bool = False) -> None:
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt, suffix = ".2f", "norm"
    else:
        fmt, suffix = "d", "raw"
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=BINARY_CLASSES, yticklabels=BINARY_CLASSES)
    plt.xlabel("Prédit"); plt.ylabel("Réel"); plt.title(f"{label} ({'norm' if normalize else 'brute'})")
    plt.tight_layout(); plt.savefig(save_dir / f"cm_{label}_{suffix}.png", dpi=110); plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    results_path = RESULTS_DIR / "results.json"
    if not results_path.exists():
        sys.exit(f"Fichier introuvable : {results_path} (lance d'abord tune_binary.py)")

    with open(results_path) as f:
        results = json.load(f)

    print(f"\n{'='*70}\n  {len(results)} modèles binaires trouvés\n{'='*70}")

    cm_dir = FIGURES_DIR / "confusion_matrices"
    cm_dir.mkdir(exist_ok=True)

    rows = []
    bench_records = []
    for r in results:
        trial_id = r["trial_id"]
        model_path = RESULTS_DIR / f"model_trial_{trial_id + 1:02d}.keras"
        label = f"T{trial_id+1:02d}"
        print(f"\n{'─'*60}\n{label}  :  {model_path.name}\n{'─'*60}")
        if not model_path.exists():
            print(f"  ⚠ manquant : {model_path}")
            continue

        model = tf.keras.models.load_model(str(model_path))
        print("  ▶ model.summary()")
        model.summary(line_length=80)
        # n_params = paramètres entraînables (compat. graphiques existants)
        n_params = sum(int(tf.size(v).numpy()) for v in model.trainable_variables)
        # Paramètres déployés = trainable + non-trainable (poids du réseau seul)
        n_deployed = model.count_params()
        # Paramètres entraînement = + état optimiseur Adam (moments m, v, step)
        n_training = sum(int(tf.size(v).numpy()) for v in model.variables)

        print("  ▶ Benchmark sur le jeu de test…")
        b = benchmark_model(model, r["config"])
        print(f"  test_acc={b['test_acc']}  test_auc={b['test_auc']}  "
              f"f1_macro={b['f1_macro']}  loss={b['test_loss']}")
        print(b["report"])

        plot_cm(np.array(b["y_true"]), np.array(b["y_pred"]), label, cm_dir, normalize=False)
        plot_cm(np.array(b["y_true"]), np.array(b["y_pred"]), label, cm_dir, normalize=True)

        row = {
            "label": label, "trial_id": trial_id, "backbone": r["config"]["backbone"],
            "test_acc": b["test_acc"], "test_loss": b["test_loss"], "test_auc": b["test_auc"],
            "precision_macro": b["precision_macro"], "recall_macro": b["recall_macro"],
            "f1_macro": b["f1_macro"], "n_params": n_params,
            "n_params_deploy": n_deployed, "n_params_train": n_training,
            "meets_target": bool(b["test_acc"] >= TARGET_ACC and b["test_loss"] <= TARGET_LOSS),
        }
        for c in BINARY_CLASSES:
            row[f"recall_{c}"] = b["recall_per_class"][c]
        rows.append(row)

        # Enregistrement au schéma commun (identique au multiclass)
        bench_records.append({
            "label": label, "trial_id": trial_id,
            "backbone": r["config"]["backbone"],
            "test_acc": b["test_acc"], "test_loss": b["test_loss"], "test_auc": b["test_auc"],
            "precision_macro": b["precision_macro"], "recall_macro": b["recall_macro"],
            "f1_macro": b["f1_macro"],
            "precision_per_class": b["precision_per_class"],
            "recall_per_class": b["recall_per_class"],
            "f1_per_class": b["f1_per_class"],
            "n_params": n_params,
            "n_params_deploy": n_deployed,
            "n_params_train": n_training,
            "meets_target": bool(b["test_acc"] >= TARGET_ACC and b["test_loss"] <= TARGET_LOSS),
            "config": r["config"],
        })
        tf.keras.backend.clear_session()

    df = pd.DataFrame(rows).sort_values("trial_id").reset_index(drop=True)

    print(f"\n{'='*70}\nTABLEAU DE SYNTHÈSE\n{'='*70}")
    cols = ["label", "backbone", "test_acc", "test_auc", "f1_macro", "n_params"]
    print(df[cols].to_string(index=False, float_format="%.4f"))

    best = df.loc[df["test_acc"].idxmax()]
    print(f"\n-> Meilleur modèle : {best['label']} ({best['backbone']})  "
          f"acc={best['test_acc']}  auc={best['test_auc']}  params={int(best['n_params']):,}")

    print(f"\n{'='*70}\nGRAPHIQUES\n{'='*70}")
    plot_bar_acc_loss_recall(df, FIGURES_DIR / "1_bar_acc_loss_recall.png")
    plot_bar_metrics(df, FIGURES_DIR / "1b_bar_acc_auc_f1.png")
    plot_acc_vs_params(df, FIGURES_DIR / "2_scatter_acc_vs_params.png")
    corr = compute_hp_importance(results)
    print("\nInfluence des hyperparamètres (|Pearson| avec test_acc) :")
    for hp, c in corr.items():
        print(f"  {hp:<22} {c:.3f}  {'█' * int(c * 30)}")
    plot_hp_importance(corr, FIGURES_DIR / "3_hp_importance.png")
    print(f"  Matrices de confusion : {cm_dir}/")

    out = RESULTS_DIR / "benchmark_results.json"
    # Sérialisation selon le schéma commun (ordre et champs identiques au multiclass)
    clean_bench = [{k: rec.get(k) for k in BENCHMARK_SCHEMA} for rec in bench_records]
    with open(out, "w") as f:
        json.dump(clean_bench, f, indent=2)
    print(f"\nMétriques sauvegardées : {out}")

    # ----------------------------------------------------------------------
    # Configs prêtes à l'emploi : meilleure accuracy & meilleur rapport perf/taille
    # (directement utilisables comme config_binary.json, epochs "production" restaurés)
    # ----------------------------------------------------------------------
    export_best_configs(bench_records)


def _prod_config(cfg: dict) -> dict:
    """Restaure les epochs 'production' (le tuning les plafonne pour aller vite)."""
    cfg = dict(cfg)
    cfg["epochs_head"] = 10
    cfg["epochs_finetune"] = 40
    return cfg


def export_best_configs(bench_records: list[dict]) -> None:
    if not bench_records:
        return

    # Meilleure accuracy (tie-break : loss la plus basse)
    best_acc = max(bench_records, key=lambda r: (r["test_acc"], -r["test_loss"]))

    # Meilleur rapport perf/taille parmi ceux qui atteignent la cible (sinon tous).
    # Taille = nb de params TOTAL avec état optimiseur Adam (n_params_train).
    pool = [r for r in bench_records if r["meets_target"]] or bench_records
    best_eff = max(pool, key=lambda r: r["test_acc"] / np.log1p(r["n_params_train"] / 1e6))

    acc_path = RESULTS_DIR / "best_acc_config.json"
    eff_path = RESULTS_DIR / "best_efficiency_config.json"
    with open(acc_path, "w") as f:
        json.dump(_prod_config(best_acc["config"]), f, indent=2)
    with open(eff_path, "w") as f:
        json.dump(_prod_config(best_eff["config"]), f, indent=2)

    print(f"\n{'='*70}\nCONFIGS EXPORTÉES (prêtes pour config_binary.json)\n{'='*70}")
    print(f"  Meilleure accuracy   : {best_acc['label']} ({best_acc['backbone']})  "
          f"acc={best_acc['test_acc']}  loss={best_acc['test_loss']}")
    print(f"    -> {acc_path}")
    print(f"  Meilleur rapport     : {best_eff['label']} ({best_eff['backbone']})  "
          f"acc={best_eff['test_acc']}  params_train={int(best_eff['n_params_train']):,}")
    print(f"    -> {eff_path}")


if __name__ == "__main__":
    main()

"""
tune_multiclass.py — Optimisation bayésienne (Optuna) du modèle multi-class 023.

Wrapper mince autour de multiclass_model.py : chaque trial génère un `cfg`, puis
réutilise EXACTEMENT le même build_model / make_dataset / run_training que le
script d'entraînement 02_model_multiclass.py (zéro duplication d'architecture).

Objectif : minimiser le nombre de paramètres tout en garantissant
  - val_accuracy >= 0.90
  - val_loss     <= 0.25
  - gap train/val <= 0.07

Sortie (compatible avec analyze_tune_results.py) :
  - tune_multiclass_results/results.json      — toutes les métriques
  - tune_multiclass_results/best_config.json  — meilleure config
  - tune_multiclass_results/comparison_*.png  — graphiques comparatifs

Remplace tune_023.py (même sortie, même format, reprise possible).

Usage :
    python3 tune_multiclass.py --trials 30
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
from tensorflow.keras import callbacks

from multiclass_model import make_dataset, run_training
from utils import CLASS_NAMES, IMG_SIZE, SEED, set_seeds

optuna.logging.set_verbosity(optuna.logging.WARNING)

RESULTS_DIR = Path("tune_multiclass_results")

TARGET_VAL_ACC  = 0.90
TARGET_VAL_LOSS = 0.25
MAX_OVERFIT_GAP = 0.07

MAX_STEP_TIME_S  = 3.0
SPEED_CHECK_STEPS = 4

_PENALTY = 1e9


# ----------------------------------------------------------------------------
# SpeedGuard : ignore les trials trop lents (spécifique au tuning)
# ----------------------------------------------------------------------------

class SpeedGuard(callbacks.Callback):
    def on_train_begin(self, logs=None):
        self._step_times: list[float] = []
        self._t0: float | None = None
        self.triggered = False

    def on_train_batch_begin(self, batch, logs=None):
        self._t0 = time.time()

    def on_train_batch_end(self, batch, logs=None):
        if self._t0 is None or batch >= SPEED_CHECK_STEPS:
            return
        self._step_times.append(time.time() - self._t0)
        if len(self._step_times) == SPEED_CHECK_STEPS:
            avg = sum(self._step_times) / SPEED_CHECK_STEPS
            if avg > MAX_STEP_TIME_S:
                print(f"\n  ⚡ SpeedGuard : moyenne step = {avg:.2f}s > {MAX_STEP_TIME_S}s — trial ignoré.")
                self.model.stop_training = True
                self.triggered = True


# ----------------------------------------------------------------------------
# Config d'un trial  (identique à tune_023.py)
# ----------------------------------------------------------------------------

def make_config_from_trial(trial: optuna.Trial) -> dict:
    n_blocs = trial.suggest_int("n_blocs", 2, 4)
    base_f  = trial.suggest_categorical("base_filters", [16, 32, 64])
    filters = [base_f * (2 ** i) for i in range(n_blocs)]

    dp_bloc = trial.suggest_float("dropout_bloc", 0.1, 0.4, step=0.05)
    return {
        "img_size":   list(IMG_SIZE),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64]),
        "seed":       SEED,
        "epochs":     30,
        "learning_rate": trial.suggest_categorical("learning_rate", [1e-3, 5e-4, 2e-4]),
        "reduce_lr_factor":   0.5,
        "reduce_lr_patience": trial.suggest_int("reduce_lr_patience", 2, 4),
        "early_stopping_patience": trial.suggest_int("early_stopping_patience", 5, 10),
        "dropout": {
            "bloc1": dp_bloc,
            "bloc2": dp_bloc,
            "bloc3": round(dp_bloc + 0.05, 2),
            "bloc4": round(dp_bloc + 0.05, 2),
            "head":  trial.suggest_float("dropout_head", 0.3, 0.6, step=0.05),
        },
        "filters":     filters,
        "dense_units": trial.suggest_categorical("dense_units", [64, 128, 256]),
        "l2":          trial.suggest_categorical("l2", [1e-4, 5e-4, 1e-3]),
        "shuffle_buffer": trial.suggest_categorical("shuffle_buffer", [300, 500, 1000]),
        "augmentation": {
            "flip":       "horizontal",
            "rotation":   trial.suggest_categorical("aug_rotation", [0.03, 0.05, 0.08]),
            "zoom":       trial.suggest_categorical("aug_zoom", [0.05, 0.1, 0.15]),
            "translation": 0.1,
        },
        "class_names": CLASS_NAMES,
    }


# ----------------------------------------------------------------------------
# Objectif Optuna
# ----------------------------------------------------------------------------

def objective(trial: optuna.Trial, all_results: list[dict]) -> float:
    cfg = make_config_from_trial(trial)
    trial_id = trial.number

    print(f"\n{'='*70}")
    print(f"TRIAL {trial_id+1} — lr={cfg['learning_rate']}  bs={cfg['batch_size']}  "
          f"filters={cfg['filters']}  dense={cfg['dense_units']}")
    print(f"{'='*70}")

    set_seeds(cfg["seed"])

    train_ds = make_dataset("train", cfg, augment=True)
    val_ds   = make_dataset("val",   cfg, augment=False)
    # NB : pas de test_ds ici. Le jeu de test ne doit JAMAIS être vu pendant le
    # tuning (sélection d'hyperparamètres) — il est réservé au benchmark final
    # dans analyze_multiclass_tune_results.py.

    speed_guard = SpeedGuard()
    model_path = RESULTS_DIR / f"model_trial_{trial_id + 1:02d}.keras"

    t0 = time.time()
    model, history = run_training(
        cfg, train_ds, val_ds,
        model_path=model_path,
        extra_callbacks=[speed_guard],
        verbose=1,
    )
    elapsed = time.time() - t0

    n_params = model.count_params()

    if speed_guard.triggered:
        tf.keras.backend.clear_session()
        print(f"  → Trial {trial_id+1} ignoré (trop lent).")
        return _PENALTY

    val_accs   = history["val_accuracy"]
    val_losses = history["val_loss"]
    train_accs = history["accuracy"]
    n_epochs   = len(val_accs)

    best_epoch    = int(np.argmin(val_losses))
    best_val_acc  = val_accs[best_epoch]
    best_val_loss = val_losses[best_epoch]
    overfit_gap   = train_accs[best_epoch] - best_val_acc
    stability     = float(np.std(val_losses[-5:])) if n_epochs >= 5 else float(np.std(val_losses))

    meets_target = (
        best_val_acc  >= TARGET_VAL_ACC and
        best_val_loss <= TARGET_VAL_LOSS and
        overfit_gap   <= MAX_OVERFIT_GAP
    )

    metrics = {
        "trial_id":          trial_id,
        "best_epoch":        best_epoch + 1,
        "n_epochs":          n_epochs,
        "best_val_acc":      round(float(best_val_acc), 4),
        "best_val_loss":     round(float(best_val_loss), 4),
        "train_acc_at_best": round(float(train_accs[best_epoch]), 4),
        "overfit_gap":       round(float(overfit_gap), 4),
        "stability":         round(stability, 4),
        "meets_target":      meets_target,
        "overfitting":       overfit_gap > MAX_OVERFIT_GAP,
        "n_params":          n_params,
        "train_time":        round(elapsed / 60, 1),
        "config":            cfg,
        "history":           {k: [round(float(v), 4) for v in vals]
                              for k, vals in history.items()},
    }
    all_results.append(metrics)

    status = "✓ CIBLE ATTEINTE" if meets_target else "✗"
    overfit_str = " ⚠ OVERFIT" if metrics["overfitting"] else ""
    print(f"\nTrial {trial_id+1} : val_acc={metrics['best_val_acc']}  "
          f"val_loss={metrics['best_val_loss']}  gap={metrics['overfit_gap']}  "
          f"params={n_params:,}  {status}{overfit_str}")

    tf.keras.backend.clear_session()

    if meets_target:
        return float(n_params)

    penalty = _PENALTY
    if best_val_acc < TARGET_VAL_ACC:
        penalty += (TARGET_VAL_ACC - best_val_acc) * 1e6
    if best_val_loss > TARGET_VAL_LOSS:
        penalty += (best_val_loss - TARGET_VAL_LOSS) * 1e6
    if overfit_gap > MAX_OVERFIT_GAP:
        penalty += (overfit_gap - MAX_OVERFIT_GAP) * 1e6
    return penalty


# ----------------------------------------------------------------------------
# Visualisations  (identiques à tune_023.py)
# ----------------------------------------------------------------------------

def make_comparison_plots(results: list[dict]) -> None:
    df = pd.DataFrame([{
        "trial":        f"T{r['trial_id']+1}",
        "val_acc":      r["best_val_acc"],
        "val_loss":     r["best_val_loss"],
        "overfit_gap":  r["overfit_gap"],
        "stability":    r["stability"],
        "n_params":     r["n_params"] / 1e6,
        "train_time":   r["train_time"],
        "meets_target": r["meets_target"],
        "overfitting":  r["overfitting"],
    } for r in results])

    colors = ["green" if m else "red" for m in df["meets_target"]]
    n = len(df)
    x = np.arange(n)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Comparaison des trials — tune_multiclass (Optuna)", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.bar(x, df["val_acc"], color=colors, alpha=0.8)
    ax.axhline(TARGET_VAL_ACC, color="green", linestyle="--", linewidth=1.5, label=f"Cible {TARGET_VAL_ACC}")
    ax.set_xticks(x); ax.set_xticklabels(df["trial"], rotation=45, ha="right")
    ax.set_title("Val Accuracy"); ax.set_ylabel("Accuracy"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.bar(x, df["val_loss"], color=colors, alpha=0.8)
    ax.axhline(TARGET_VAL_LOSS, color="orange", linestyle="--", linewidth=1.5, label=f"Cible {TARGET_VAL_LOSS}")
    ax.set_xticks(x); ax.set_xticklabels(df["trial"], rotation=45, ha="right")
    ax.set_title("Val Loss"); ax.set_ylabel("Loss"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    gap_colors = ["red" if o else "steelblue" for o in df["overfitting"]]
    ax.bar(x, df["overfit_gap"], color=gap_colors, alpha=0.8)
    ax.axhline(MAX_OVERFIT_GAP, color="red", linestyle="--", linewidth=1.5, label=f"Seuil {MAX_OVERFIT_GAP}")
    ax.set_xticks(x); ax.set_xticklabels(df["trial"], rotation=45, ha="right")
    ax.set_title("Gap Train/Val (overfitting)"); ax.set_ylabel("Gap"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.bar(x, df["stability"], color=colors, alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(df["trial"], rotation=45, ha="right")
    ax.set_title("Stabilité (std val_loss, 5 dern.)"); ax.set_ylabel("std"); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.bar(x, df["n_params"], color="steelblue", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(df["trial"], rotation=45, ha="right")
    ax.set_title("Nombre de paramètres (M)"); ax.set_ylabel("Params (M)"); ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    sc = ax.scatter(df["n_params"], df["val_acc"], c=df["overfit_gap"],
                    cmap="RdYlGn_r", s=100, vmin=0, vmax=0.15)
    plt.colorbar(sc, ax=ax, label="Overfit gap")
    for _, row in df.iterrows():
        ax.annotate(row["trial"], (row["n_params"], row["val_acc"]),
                    textcoords="offset points", xytext=(5, 3), fontsize=8)
    ax.axhline(TARGET_VAL_ACC, color="green", linestyle="--", linewidth=1)
    ax.set_title("Params (M) vs Val Accuracy"); ax.set_xlabel("Params (M)"); ax.set_ylabel("Val Acc")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "comparison_overview.png", dpi=120)
    plt.close()

    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)
    for i, r in enumerate(results):
        h = r["history"]
        ep = range(1, len(h["loss"]) + 1)
        c = "green" if r["meets_target"] else ("orange" if not r["overfitting"] else "red")
        axes[0, i].plot(ep, h["loss"], label="train")
        axes[0, i].plot(ep, h["val_loss"], label="val")
        axes[0, i].axhline(TARGET_VAL_LOSS, color="orange", linestyle="--", alpha=0.5)
        axes[0, i].set_title(f"T{r['trial_id']+1} Loss", color=c)
        axes[0, i].legend(fontsize=7); axes[0, i].grid(True, alpha=0.3)
        axes[1, i].plot(ep, h["accuracy"], label="train")
        axes[1, i].plot(ep, h["val_accuracy"], label="val")
        axes[1, i].axhline(TARGET_VAL_ACC, color="green", linestyle="--", alpha=0.5)
        axes[1, i].set_title(f"T{r['trial_id']+1} Acc", color=c)
        axes[1, i].legend(fontsize=7); axes[1, i].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "comparison_curves.png", dpi=100)
    plt.close()
    print(f"\nGraphiques : {RESULTS_DIR}/comparison_*.png")


def print_summary_table(results: list[dict], best_idx: int) -> None:
    print("\n" + "=" * 95)
    print("RÉSUMÉ DES TRIALS")
    print("=" * 95)
    rows = []
    for r in results:
        rows.append({
            "Trial":     f"T{r['trial_id']+1}",
            "val_acc":   r["best_val_acc"],
            "val_loss":  r["best_val_loss"],
            "gap":       r["overfit_gap"],
            "params(k)": r["n_params"] // 1000,
            "time(min)": r["train_time"],
            "cible":     "✓" if r["meets_target"] else "✗",
            "overfit":   "⚠" if r["overfitting"] else "ok",
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format="%.4f"))
    best = results[best_idx]
    print(f"\n→ Meilleur trial : T{best['trial_id']+1} "
          f"(val_acc={best['best_val_acc']}, val_loss={best['best_val_loss']}, "
          f"params={best['n_params']:,})")
    print(f"  Config sauvegardée dans : {RESULTS_DIR}/best_config.json")


# ----------------------------------------------------------------------------
# Main  (resume + TPE + importance, identiques à tune_023.py)
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=6, help="Nombre de trials")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    results_path = RESULTS_DIR / "results.json"

    all_results: list[dict] = []
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
        print(f"Reprise : {len(all_results)} trials déjà effectués")

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        study_name="tune_multiclass",
    )

    # Réinjecte les trials connus pour que Optuna en tienne compte
    for r in all_results:
        cfg = r["config"]
        params = {
            "n_blocs":       len(cfg["filters"]),
            "base_filters":  cfg["filters"][0],
            "batch_size":    cfg["batch_size"],
            "learning_rate": cfg["learning_rate"],
            "reduce_lr_patience":       cfg["reduce_lr_patience"],
            "early_stopping_patience":  cfg["early_stopping_patience"],
            "dropout_bloc":  cfg["dropout"]["bloc1"],
            "dropout_head":  cfg["dropout"]["head"],
            "dense_units":   cfg["dense_units"],
            "l2":            cfg["l2"],
            "shuffle_buffer": cfg["shuffle_buffer"],
            "aug_rotation":  cfg["augmentation"]["rotation"],
            "aug_zoom":      cfg["augmentation"]["zoom"],
        }
        value = float(r["n_params"]) if r["meets_target"] else _PENALTY
        study.add_trial(optuna.trial.create_trial(
            params=params,
            distributions={
                "n_blocs":       optuna.distributions.IntDistribution(2, 4),
                "base_filters":  optuna.distributions.CategoricalDistribution([16, 32, 64]),
                "batch_size":    optuna.distributions.CategoricalDistribution([32, 64]),
                "learning_rate": optuna.distributions.CategoricalDistribution([1e-3, 5e-4, 2e-4]),
                "reduce_lr_patience":      optuna.distributions.IntDistribution(2, 4),
                "early_stopping_patience": optuna.distributions.IntDistribution(5, 10),
                "dropout_bloc":  optuna.distributions.FloatDistribution(0.1, 0.4, step=0.05),
                "dropout_head":  optuna.distributions.FloatDistribution(0.3, 0.6, step=0.05),
                "dense_units":   optuna.distributions.CategoricalDistribution([64, 128, 256]),
                "l2":            optuna.distributions.CategoricalDistribution([1e-4, 5e-4, 1e-3]),
                "shuffle_buffer": optuna.distributions.CategoricalDistribution([300, 500, 1000]),
                "aug_rotation":  optuna.distributions.CategoricalDistribution([0.03, 0.05, 0.08]),
                "aug_zoom":      optuna.distributions.CategoricalDistribution([0.05, 0.1, 0.15]),
            },
            value=value,
        ))

    study.optimize(
        lambda trial: objective(trial, all_results),
        n_trials=args.trials,
        callbacks=[lambda study, trial: (
            open(results_path, "w").write(json.dumps(all_results, indent=2))
        )],
    )

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    if not all_results:
        print("Aucun trial valide.")
        return

    def score(r):
        return (int(r["meets_target"]), -r["n_params"])

    best_idx = max(range(len(all_results)), key=lambda i: score(all_results[i]))
    best_cfg = all_results[best_idx]["config"]

    with open(RESULTS_DIR / "best_config.json", "w") as f:
        json.dump(best_cfg, f, indent=2)

    make_comparison_plots(all_results)
    print_summary_table(all_results, best_idx)

    try:
        importances = optuna.importance.get_param_importances(study)
        print("\n" + "=" * 50)
        print("IMPORTANCE DES HYPERPARAMÈTRES")
        print("=" * 50)
        for param, importance in importances.items():
            bar = "█" * int(importance * 40)
            print(f"  {param:<30} {importance:.3f}  {bar}")
    except Exception as e:
        print(f"\n(Importance non calculable : {e})")


if __name__ == "__main__":
    main()

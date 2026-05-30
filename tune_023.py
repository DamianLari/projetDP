"""
tune_023.py — Optimisation bayésienne des hyperparamètres (Optuna) pour le modèle 023.

Objectif principal : minimiser le nombre de paramètres tout en garantissant :
  - val_accuracy >= 0.90
  - val_loss     <= 0.25
  - gap train/val <= 0.07

Optuna utilise un sampler TPE (Tree-structured Parzen Estimator) : après chaque
trial il apprend quelles valeurs sont prometteuses et oriente les suivants.

Sortie :
  - tune_results/results.json      — toutes les métriques
  - tune_results/best_config.json  — meilleure config
  - tune_results/comparison_*.png  — graphiques comparatifs

Usage :
    python3 tune_023.py --trials 30
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
from tensorflow.keras import callbacks, layers, models, regularizers

from utils import (
    BATCH_SIZE, CFG, CLASS_NAMES, HISTORIES_DIR, IMG_SIZE, MODELS_DIR, SEED,
    SPLIT_DIR, save_history, set_seeds,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

RESULTS_DIR = Path("tune_results")
IMG_H, IMG_W = IMG_SIZE
AUTOTUNE = tf.data.AUTOTUNE
N_CLASSES = len(CLASS_NAMES)

TARGET_VAL_ACC  = 0.90
TARGET_VAL_LOSS = 0.25
MAX_OVERFIT_GAP = 0.07

MAX_STEP_TIME_S  = 3.0
SPEED_CHECK_STEPS = 4

# Pénalité retournée à Optuna quand un trial ne mérite pas d'être évalué
_PENALTY = 1e9


# ----------------------------------------------------------------------------
# Pipeline tf.data
# ----------------------------------------------------------------------------

def _decode_img(file_path: tf.Tensor) -> tf.Tensor:
    raw = tf.io.read_file(file_path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_H, IMG_W])
    return tf.cast(img, tf.float32) / 255.0


def _get_label(file_path: tf.Tensor) -> tf.Tensor:
    parts = tf.strings.split(file_path, "/")
    n = tf.shape(parts)[0]
    class_name = parts[n - 2]
    matches = tf.cast(tf.equal(class_name, tf.constant(CLASS_NAMES)), tf.int32)
    return tf.cast(tf.argmax(matches), tf.int32)


def _make_augment_fn(aug: dict):
    def _augment(img):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.1)
        img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
        img = tf.image.random_saturation(img, lower=0.9, upper=1.1)
        shape = tf.shape(img)
        crop_size = tf.cast(tf.cast(shape[:2], tf.float32) * (1 - aug["zoom"]), tf.int32)
        img = tf.image.random_crop(img, size=[crop_size[0], crop_size[1], 3])
        img = tf.image.resize(img, [IMG_H, IMG_W])
        return tf.clip_by_value(img, 0.0, 1.0)
    return _augment


def make_dataset(subset: str, cfg: dict, augment: bool = False) -> tf.data.Dataset:
    pattern = str(SPLIT_DIR / subset / "*" / "*")
    ds = tf.data.Dataset.list_files(pattern, shuffle=(subset == "train"), seed=cfg["seed"])
    ds = ds.map(lambda p: (_decode_img(p), _get_label(p)), num_parallel_calls=AUTOTUNE)
    if augment:
        aug_fn = _make_augment_fn(cfg["augmentation"])
        ds = ds.map(lambda img, lbl: (aug_fn(img), lbl), num_parallel_calls=AUTOTUNE)
        ds = ds.shuffle(cfg["shuffle_buffer"], seed=cfg["seed"], reshuffle_each_iteration=True)
    return ds.batch(cfg["batch_size"]).prefetch(AUTOTUNE)


# ----------------------------------------------------------------------------
# Modèle
# ----------------------------------------------------------------------------

def build_model(cfg: dict) -> models.Model:
    dp = cfg["dropout"]
    filters = cfg["filters"]
    dropout_keys = ["bloc1", "bloc2", "bloc3", "bloc4"]

    inp = layers.Input(shape=(IMG_H, IMG_W, 3), name="input_image")
    x = inp
    for i, f in enumerate(filters):
        d_key = dropout_keys[min(i, 3)]
        x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        if i < 2:
            x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
            x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(dp[d_key])(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(cfg["dense_units"], activation="relu",
                     kernel_regularizer=regularizers.l2(cfg["l2"]))(x)
    x = layers.Dropout(dp["head"])(x)
    out = layers.Dense(N_CLASSES, activation="softmax", name="predictions")(x)
    return models.Model(inp, out, name="cnn_trial")


# ----------------------------------------------------------------------------
# SpeedGuard callback
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
# Fonction objectif Optuna
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
    test_ds  = make_dataset("test",  cfg, augment=False)

    model = build_model(cfg)
    n_params = model.count_params()
    print(f"Params : {n_params:,}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg["learning_rate"]),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    model_path = RESULTS_DIR / f"model_trial_{trial_id:02d}.keras"
    speed_guard = SpeedGuard()
    cb = [
        speed_guard,
        callbacks.EarlyStopping(monitor="val_loss",
                                patience=cfg["early_stopping_patience"],
                                restore_best_weights=True, verbose=0),
        callbacks.ReduceLROnPlateau(monitor="val_loss",
                                    factor=cfg["reduce_lr_factor"],
                                    patience=cfg["reduce_lr_patience"],
                                    min_lr=1e-6, verbose=0),
        callbacks.ModelCheckpoint(str(model_path), monitor="val_loss",
                                  save_best_only=True, verbose=0),
    ]

    t0 = time.time()
    hist = model.fit(train_ds, validation_data=val_ds,
                     epochs=cfg["epochs"], callbacks=cb, verbose=1)
    elapsed = time.time() - t0

    if speed_guard.triggered:
        tf.keras.backend.clear_session()
        print(f"  → Trial {trial_id+1} ignoré (trop lent).")
        return _PENALTY

    # Métriques
    val_accs   = hist.history["val_accuracy"]
    val_losses = hist.history["val_loss"]
    train_accs = hist.history["accuracy"]
    n_epochs   = len(val_accs)

    best_epoch      = int(np.argmin(val_losses))
    best_val_acc    = val_accs[best_epoch]
    best_val_loss   = val_losses[best_epoch]
    overfit_gap     = train_accs[best_epoch] - best_val_acc
    stability       = float(np.std(val_losses[-5:])) if n_epochs >= 5 else float(np.std(val_losses))

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)

    meets_target = (
        best_val_acc  >= TARGET_VAL_ACC and
        best_val_loss <= TARGET_VAL_LOSS and
        overfit_gap   <= MAX_OVERFIT_GAP
    )

    metrics = {
        "trial_id":        trial_id,
        "best_epoch":      best_epoch + 1,
        "n_epochs":        n_epochs,
        "best_val_acc":    round(float(best_val_acc), 4),
        "best_val_loss":   round(float(best_val_loss), 4),
        "train_acc_at_best": round(float(train_accs[best_epoch]), 4),
        "overfit_gap":     round(float(overfit_gap), 4),
        "stability":       round(stability, 4),
        "test_acc":        round(float(test_acc), 4),
        "test_loss":       round(float(test_loss), 4),
        "meets_target":    meets_target,
        "overfitting":     overfit_gap > MAX_OVERFIT_GAP,
        "n_params":        n_params,
        "train_time":      round(elapsed / 60, 1),
        "config":          cfg,
        "history":         {k: [round(float(v), 4) for v in vals]
                            for k, vals in hist.history.items()},
    }
    all_results.append(metrics)

    status = "✓ CIBLE ATTEINTE" if meets_target else "✗"
    overfit_str = " ⚠ OVERFIT" if metrics["overfitting"] else ""
    print(f"\nTrial {trial_id+1} : val_acc={metrics['best_val_acc']}  "
          f"val_loss={metrics['best_val_loss']}  gap={metrics['overfit_gap']}  "
          f"params={n_params:,}  {status}{overfit_str}")

    tf.keras.backend.clear_session()

    # Optuna minimise : on retourne n_params si cible atteinte, sinon une pénalité
    if meets_target:
        return float(n_params)

    # Pénalité proportionnelle à l'écart aux objectifs (guide Optuna vers la cible)
    penalty = _PENALTY
    if best_val_acc < TARGET_VAL_ACC:
        penalty += (TARGET_VAL_ACC - best_val_acc) * 1e6
    if best_val_loss > TARGET_VAL_LOSS:
        penalty += (best_val_loss - TARGET_VAL_LOSS) * 1e6
    if overfit_gap > MAX_OVERFIT_GAP:
        penalty += (overfit_gap - MAX_OVERFIT_GAP) * 1e6
    return penalty


# ----------------------------------------------------------------------------
# Visualisations
# ----------------------------------------------------------------------------

def make_comparison_plots(results: list[dict]) -> None:
    df = pd.DataFrame([{
        "trial":        f"T{r['trial_id']+1}",
        "val_acc":      r["best_val_acc"],
        "val_loss":     r["best_val_loss"],
        "test_acc":     r["test_acc"],
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
    fig.suptitle("Comparaison des trials — tune_023 (Optuna)", fontsize=14, fontweight="bold")

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
    ax.bar(x, df["test_acc"], color=colors, alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(df["trial"], rotation=45, ha="right")
    ax.set_title("Test Accuracy"); ax.set_ylabel("Accuracy"); ax.grid(True, alpha=0.3)

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
            "test_acc":  r["test_acc"],
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
# Main
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
        study_name="tune_023",
    )

    # Injecte les trials déjà connus dans l'étude pour que Optuna en tienne compte
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

    # Sauvegarde finale
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    if not all_results:
        print("Aucun trial valide.")
        return

    # Meilleure config : cible atteinte + moins de params
    def score(r):
        return (int(r["meets_target"]), -r["n_params"])

    best_idx = max(range(len(all_results)), key=lambda i: score(all_results[i]))
    best_cfg = all_results[best_idx]["config"]

    with open(RESULTS_DIR / "best_config.json", "w") as f:
        json.dump(best_cfg, f, indent=2)

    make_comparison_plots(all_results)
    print_summary_table(all_results, best_idx)

    # Importance des hyperparamètres
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
